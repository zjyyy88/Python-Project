#!/usr/bin/env python3
"""
基于 XGBoost 的水分子吸附能描述符挖掘脚本。

核心流程：
1) 将元素标签转换为具有物理意义的元素描述符。
2) 用留一交叉验证（LOOCV）训练并评估 XGBoost（适合小样本）。
3) 使用 SHAP（若可用）或置换重要性解释描述符贡献。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from ase.data import atomic_masses, atomic_numbers, covalent_radii, vdw_radii
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    shap = None


sns.set_theme(style="whitegrid")


def parse_element_label(label: str) -> tuple[str, int]:
    """
    将 "Fe3+"、"Ni2+"、"Co" 这类标签解析为 (元素符号, 氧化态)。
    """
    # 将输入标签标准化为紧凑字符串。
    text = str(label).strip()
    # 分组1: 元素符号；分组2: 可选数字；分组3: 可选号。
    match = re.fullmatch(r"([A-Z][a-z]?)(\d+)?([+-])?", text)
    if not match:
        raise ValueError(f"Cannot parse element label: {label}")

    symbol, number_text, sign = match.groups()
    oxidation_state = 0
    if number_text and sign:
        # 典型氧化态标签，如 Fe3+、O2-。
        sign_factor = 1 if sign == "+" else -1
        oxidation_state = sign_factor * int(number_text)
    elif number_text and not sign:
        # 保守回退：若只有数字无符号，则按氧化态处理。
        oxidation_state = int(number_text)

    return symbol, oxidation_state


def build_element_descriptors(symbol: str, oxidation_state: int) -> dict[str, float]:
    """
    基于 ASE 周期表数据构造元素描述符。
    """
    if symbol not in atomic_numbers:
        raise ValueError(f"Unknown element symbol: {symbol}")

    # 从 ASE 周期表中提取原子性质。
    z = int(atomic_numbers[symbol])
    mass = float(atomic_masses[z])
    cov_radius = float(covalent_radii[z]) if z < len(covalent_radii) else np.nan
    vdw_radius = float(vdw_radii[z]) if z < len(vdw_radii) else np.nan

    if np.isnan(vdw_radius):
        vdw_radius = np.nan

    # 用轻量家族标签表示粗粒度化学类别信息。
    is_lanthanide = 1.0 if 57 <= z <= 71 else 0.0
    is_actinide = 1.0 if 89 <= z <= 103 else 0.0
    is_transition_metal = 1.0 if (21 <= z <= 30) or (39 <= z <= 48) or (72 <= z <= 80) else 0.0

    return {
        "atomic_number": float(z),
        "atomic_mass": mass,
        "covalent_radius": cov_radius,
        "vdw_radius": vdw_radius,
        "oxidation_state": float(oxidation_state),
        "is_transition_metal": is_transition_metal,
        "is_lanthanide": is_lanthanide,
        "is_actinide": is_actinide,
    }


def prepare_dataset(
    df: pd.DataFrame,
    element_col: str,
    dband_col: str,
    target_col: str,
    use_element_onehot: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    将原始数据转换为模型特征 X 与目标 y。
    """
    # 提前校验必需列，避免后续静默错配。
    needed = {element_col, dband_col, target_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # 将每一行元素标签扩展为对应描述符。
    records = []
    base_symbols = []
    for raw_label in df[element_col]:
        symbol, oxidation = parse_element_label(raw_label)
        base_symbols.append(symbol)
        records.append(build_element_descriptors(symbol, oxidation))

    # 保留原始列，并追加解析后的描述符列，便于追踪。
    descriptor_df = pd.DataFrame(records)
    out_df = df.copy()
    out_df["element_symbol"] = base_symbols
    out_df = pd.concat([out_df.reset_index(drop=True), descriptor_df], axis=1)

    # 强制转换目标列与 d-band 列为数值。
    out_df["d_band"] = pd.to_numeric(out_df[dband_col], errors="coerce")
    y = pd.to_numeric(out_df[target_col], errors="coerce")

    # 物理驱动的核心特征集合。
    feature_cols = [
        "d_band",
        "atomic_number",
        "atomic_mass",
        "covalent_radius",
        "vdw_radius",
        "oxidation_state",
        "is_transition_metal",
        "is_lanthanide",
        "is_actinide",
    ]

    X = out_df[feature_cols].copy()

    if use_element_onehot:
        # 可选元素身份 one-hot 特征：当元素类别本身有额外信息时有帮助。
        dummies = pd.get_dummies(out_df["element_symbol"], prefix="el", dtype=float)
        X = pd.concat([X, dummies], axis=1)

    # 填补 NaN（例如某些元素在表中缺少 vdw 半径）。
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    valid = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    prepared_df = out_df.loc[valid].reset_index(drop=True)

    # LOOCV 虽可用于小样本，但样本过少时结果通常不稳定。
    if len(X) < 8:
        raise ValueError("Too few valid rows after preprocessing (<8).")

    return X, y, prepared_df


def build_model(random_state: int) -> xgb.XGBRegressor:
    """
    针对小样本吸附数据设置的 XGBoost 模型。
    """
    # 使用相对保守的树深与正则化以降低过拟合风险。
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.1,
        min_child_weight=1,
        random_state=random_state,
        n_jobs=-1,
    )


def run_loocv(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[np.ndarray, dict[str, float]]:
    """
    使用留一交叉验证（LOOCV）评估小样本性能。
    """
    # 每次用 N-1 个样本训练，并预测被留出的 1 个样本。
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in loo.split(X):
        model = build_model(random_state=random_state)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        y_pred[test_idx[0]] = float(model.predict(X.iloc[test_idx])[0])

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return y_pred, metrics


def compute_importance(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """
    汇总 gain/置换/（可选）SHAP 三类特征重要性。
    """
    # Gain 重要性：来自模型内部的分裂贡献。
    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "gain_importance": model.feature_importances_,
        }
    )

    # 置换重要性：打乱单个特征后，观察性能下降幅度。
    perm = permutation_importance(
        model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=120,
        random_state=random_state,
        n_jobs=-1,
    )
    importance_df["perm_importance"] = perm.importances_mean

    if SHAP_AVAILABLE:
        # SHAP 给出局部加性归因，均值绝对值可作为全局排序指标。
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        importance_df["shap_mean_abs"] = np.mean(np.abs(shap_values), axis=0)
        importance_df = importance_df.sort_values("shap_mean_abs", ascending=False)
    else:
        # SHAP 不可用时，回退到置换重要性排序。
        importance_df["shap_mean_abs"] = np.nan
        importance_df = importance_df.sort_values("perm_importance", ascending=False)

    return importance_df.reset_index(drop=True)


def plot_outputs(
    y_true: pd.Series,
    y_pred: np.ndarray,
    importance_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    保存一致性图与特征重要性图。
    """
    # 一致性图：点越接近 y=x，预测越准确。
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=60, alpha=0.85)
    lim_low = min(float(np.min(y_true)), float(np.min(y_pred))) - 0.05
    lim_high = max(float(np.max(y_true)), float(np.max(y_pred))) + 0.05
    plt.plot([lim_low, lim_high], [lim_low, lim_high], "r--", lw=2)
    plt.xlim(lim_low, lim_high)
    plt.ylim(lim_low, lim_high)
    plt.xlabel("True adsorption energy (eV)")
    plt.ylabel("LOOCV predicted adsorption energy (eV)")
    plt.title("LOOCV Parity Plot")
    plt.tight_layout()
    plt.savefig(output_dir / "loocv_parity.png", dpi=300)
    plt.close()

    # 重要性图：若 SHAP 可用，优先展示 SHAP 排序。
    if importance_df["shap_mean_abs"].notna().any():
        rank_col = "shap_mean_abs"
        title = "Feature importance (mean |SHAP|)"
    else:
        rank_col = "perm_importance"
        title = "Feature importance (permutation)"

    top_df = importance_df.head(12).sort_values(rank_col, ascending=True)

    # 横向柱状图，按重要性由低到高绘制，便于读取排名。
    plt.figure(figsize=(8, 6))
    plt.barh(top_df["feature"], top_df[rank_col])
    plt.xlabel(rank_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / "descriptor_importance.png", dpi=300)
    plt.close()


def save_summary(
    output_dir: Path,
    input_path: Path,
    metrics: dict[str, float],
    importance_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """
    保存模型质量与描述符排序的简要文本报告。
    """
    # 输出高层摘要，便于快速查看与分享。
    top5 = importance_df.head(5)
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("XGBoost descriptor analysis for adsorption energy\n")
        f.write("=" * 56 + "\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Samples: LOOCV on {len(feature_names)} features\n")
        f.write(f"SHAP available: {SHAP_AVAILABLE}\n\n")

        f.write("LOOCV metrics\n")
        f.write(f"  MAE  = {metrics['mae']:.4f} eV\n")
        f.write(f"  RMSE = {metrics['rmse']:.4f} eV\n")
        f.write(f"  R2   = {metrics['r2']:.4f}\n\n")

        f.write("Top descriptors\n")
        for _, row in top5.iterrows():
            if SHAP_AVAILABLE and not np.isnan(row["shap_mean_abs"]):
                score_text = f"SHAP={row['shap_mean_abs']:.5f}"
            else:
                score_text = f"Permutation={row['perm_importance']:.5f}"
            f.write(f"  - {row['feature']}: {score_text}\n")


def parse_args() -> argparse.Namespace:
    # 默认输入/输出路径基于脚本目录，便于跨环境复用。
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Find descriptors of water adsorption energy using XGBoost"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "dband_adsorption_data.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "xgboost_descriptor_results",
        help="Directory for outputs",
    )
    parser.add_argument("--element-col", default="Element", help="Element label column")
    parser.add_argument("--dband-col", default="d-band", help="d-band descriptor column")
    parser.add_argument("--target-col", default="Eads", help="Target adsorption energy column")
    parser.add_argument(
        "--use-element-onehot",
        action="store_true",
        help="Add one-hot element identity features in addition to physical descriptors",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    # 1) 解析命令行参数并准备输出目录。
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2) 读取原始数据并构建可用于建模的特征矩阵。
    df = pd.read_csv(args.input)
    X, y, prepared_df = prepare_dataset(
        df=df,
        element_col=args.element_col,
        dband_col=args.dband_col,
        target_col=args.target_col,
        use_element_onehot=args.use_element_onehot,
    )

    # 3) 用 LOOCV 做评估（适合小样本）。
    y_pred, metrics = run_loocv(X, y, random_state=args.random_state)

    # 4) 在全量数据上训练最终模型，用于计算特征重要性。
    model = build_model(random_state=args.random_state)
    model.fit(X, y, verbose=False)

    importance_df = compute_importance(model, X, y, random_state=args.random_state)

    pred_df = prepared_df.copy()
    pred_df["y_true"] = y
    pred_df["y_pred_loocv"] = y_pred
    pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred_loocv"])

    # 5) 保存表格结果与图像结果。
    prepared_df.to_csv(args.output_dir / "prepared_data.csv", index=False)
    pred_df.to_csv(args.output_dir / "loocv_predictions.csv", index=False)
    importance_df.to_csv(args.output_dir / "descriptor_importance.csv", index=False)

    plot_outputs(y_true=y, y_pred=y_pred, importance_df=importance_df, output_dir=args.output_dir)
    save_summary(
        output_dir=args.output_dir,
        input_path=args.input,
        metrics=metrics,
        importance_df=importance_df,
        feature_names=list(X.columns),
    )

    # 6) 在终端输出简洁运行摘要。
    print("=" * 64)
    print("XGBoost descriptor analysis finished")
    print(f"Input: {args.input}")
    print(f"Rows used: {len(X)}")
    print(f"Features used: {len(X.columns)}")
    print(f"LOOCV MAE:  {metrics['mae']:.4f} eV")
    print(f"LOOCV RMSE: {metrics['rmse']:.4f} eV")
    print(f"LOOCV R2:   {metrics['r2']:.4f}")

    print("Top descriptors:")
    for _, row in importance_df.head(5).iterrows():
        if SHAP_AVAILABLE and not np.isnan(row["shap_mean_abs"]):
            score = row["shap_mean_abs"]
            score_name = "SHAP"
        else:
            score = row["perm_importance"]
            score_name = "Permutation"
        print(f"  {row['feature']:<22} {score_name}={score:.5f}")

    print(f"Results saved to: {args.output_dir}")
    if not SHAP_AVAILABLE:
        print("SHAP is not installed. Script used permutation importance as fallback.")


if __name__ == "__main__":
    main()
