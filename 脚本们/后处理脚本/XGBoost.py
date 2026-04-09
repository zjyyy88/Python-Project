#!/usr/bin/env python3
"""
水分子吸附能预测与描述符分析
使用XGBoost从表面性质预测水分子吸附能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# 机器学习相关库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 可视化设置
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class WaterAdsorptionAnalyzer:
    """水分子吸附能分析器"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.features = None
        self.model_features = None
        self.target = None
        self.target_column = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.feature_importance = None
        
    def load_and_prepare_data(self, target_column='adsorption_energy'):
        """
        加载和准备数据
        
        参数:
        target_column: 目标变量列名，通常是吸附能
        """
        print("📥 加载数据...")
        
        if self.data_path:
            data_path_text = str(self.data_path).lower()
            if data_path_text.endswith('.csv'):
                self.df = pd.read_csv(self.data_path)
            elif data_path_text.endswith('.xlsx'):
                self.df = pd.read_excel(self.data_path)
            else:
                raise ValueError("仅支持 .csv 或 .xlsx 数据文件")
        else:
            # 示例数据，实际使用时替换
            self._create_sample_data()
        
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {list(self.df.columns)}")
        
        # 分离特征和目标
        if target_column not in self.df.columns:
            fallback_candidates = ['adsorption_energy', 'Eads', 'eads']
            fallback_target = next((c for c in fallback_candidates if c in self.df.columns), None)
            if fallback_target is None:
                raise ValueError(
                    f"目标列 '{target_column}' 不在数据中。可用列: {list(self.df.columns)}"
                )
            print(f"⚠️ 目标列 '{target_column}' 不存在，自动使用 '{fallback_target}'")
            target_column = fallback_target
        
        self.target_column = target_column
        self.target = self.df[target_column]
        self.features = self.df.drop(columns=[target_column])
        
        # 处理缺失值
        self._handle_missing_values()
        
        # 分离分类特征和数值特征
        self._identify_feature_types()
        
        return self.df
    
    def _create_sample_data(self):
        """创建示例数据（如果无真实数据）"""
        print("⚠️ 无真实数据，创建示例数据...")
        
        np.random.seed(42)
        n_samples = 200
        
        # 生成描述符特征（示例）
        data = {
            'surface_element': np.random.choice(['Li', 'Na', 'K', 'Mg', 'Ca', 'Al', 'Si'], n_samples),
            'atomic_radius': np.random.uniform(1.0, 3.0, n_samples),
            'electronegativity': np.random.uniform(0.5, 3.5, n_samples),
            'ionization_energy': np.random.uniform(3.0, 10.0, n_samples),
            'electron_affinity': np.random.uniform(0.0, 3.0, n_samples),
            'valence_electrons': np.random.randint(1, 7, n_samples),
            'surface_energy': np.random.uniform(0.5, 3.0, n_samples),
            'work_function': np.random.uniform(3.0, 6.0, n_samples),
            'band_gap': np.random.uniform(0.0, 5.0, n_samples),
            'd_band_center': np.random.uniform(-3.0, 0.0, n_samples),
            'coordination_number': np.random.randint(1, 12, n_samples),
            'lattice_constant': np.random.uniform(3.0, 6.0, n_samples),
            'surface_area': np.random.uniform(10.0, 50.0, n_samples),
            'water_distance': np.random.uniform(1.5, 3.0, n_samples),
            'water_angle': np.random.uniform(100.0, 120.0, n_samples),
        }
        
        # 计算模拟吸附能（基于一些物理关系）
        data['adsorption_energy'] = (
            -0.5 * data['electronegativity'] +
            0.3 * data['atomic_radius'] -
            0.4 * data['work_function'] +
            0.2 * data['d_band_center'] +
            np.random.normal(0, 0.1, n_samples)
        )
        
        self.df = pd.DataFrame(data)
        
        # 保存示例数据
        self.df.to_csv('sample_water_adsorption_data.csv', index=False)
        print("✅ 示例数据已保存: sample_water_adsorption_data.csv")
    
    def _handle_missing_values(self):
        """处理缺失值"""
        missing_percent = self.df.isnull().sum() / len(self.df) * 100
        
        print("\n🔍 缺失值分析:")
        missing_df = pd.DataFrame({
            '缺失百分比': missing_percent[missing_percent > 0]
        })
        if not missing_df.empty:
            print(missing_df)
            
            # 填充缺失值
            for col in self.df.columns:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            print("✅ 缺失值已处理")
        else:
            print("✓ 无缺失值")
    
    def _identify_feature_types(self):
        """识别特征类型"""
        self.numeric_features = self.features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"\n📊 特征统计:")
        print(f"数值特征 ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"分类特征 ({len(self.categorical_features)}): {self.categorical_features}")
        print(f"目标变量: 吸附能 (范围: {self.target.min():.3f} 到 {self.target.max():.3f} eV)")
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """数据预处理"""
        print(f"\n⚙️ 数据预处理...")
        
        # 处理分类变量
        if self.categorical_features:
            self.features = pd.get_dummies(
                self.features, 
                columns=self.categorical_features,
                drop_first=True
            )

        # 保留未标准化特征，便于后续可解释性分析
        self.model_features = self.features.copy()

        # 先划分再标准化，避免数据泄漏
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            self.model_features, self.target,
            test_size=test_size,
            random_state=random_state
        )
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train_raw),
            columns=X_train_raw.columns,
            index=X_train_raw.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(X_test_raw),
            columns=X_test_raw.columns,
            index=X_test_raw.index
        )
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _build_default_model(self, early_stopping_rounds=50):
        """构建默认XGBoost模型（兼容xgboost 3.x）。"""
        return xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
            early_stopping_rounds=early_stopping_rounds
        )

    def _get_cv_folds(self, max_cv=5):
        """按训练样本数自动确定交叉验证折数。"""
        n_train = 0 if self.y_train is None else len(self.y_train)
        if n_train < 2:
            return 0
        return min(max_cv, n_train)
    
    def train_xgboost_model(self, use_grid_search=False):
        """训练XGBoost模型"""
        print(f"\n🤖 训练XGBoost模型...")
        
        if use_grid_search:
            model = self._grid_search()
        else:
            model = self._build_default_model(early_stopping_rounds=50)
            
            # 训练模型
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                verbose=False
            )
        
        self.model = model
        
        # 评估模型
        self._evaluate_model()
        
        # 获取特征重要性
        self._get_feature_importance()
        
        return model
    
    def _grid_search(self):
        """网格搜索优化超参数"""
        print("🔍 执行网格搜索优化...")

        n_train = len(self.y_train)
        if n_train < 10:
            print("⚠️ 训练样本少于10，不建议网格搜索，自动切换为默认参数训练。")
            model = self._build_default_model(early_stopping_rounds=20)
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                verbose=False
            )
            return model

        cv_folds = self._get_cv_folds(max_cv=5)
        if cv_folds < 2:
            raise ValueError(f"训练样本过少({n_train})，无法执行交叉验证")
        
        # 定义参数网格
        if n_train < 50:
            param_grid = {
                'n_estimators': [100, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [2, 3, 4],
                'min_child_weight': [1, 2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
            }
        else:
            param_grid = {
                'n_estimators': [200, 500],
                'learning_rate': [0.03, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
            }

        candidate_count = int(np.prod([len(v) for v in param_grid.values()]))
        print(f"网格规模: {candidate_count} 组参数, cv={cv_folds}")
        
        # 基础模型
        xgb_model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        
        # 网格搜索
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳分数: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self):
        """评估模型性能"""
        # 训练集预测
        y_train_pred = self.model.predict(self.X_train)
        
        # 测试集预测
        y_test_pred = self.model.predict(self.X_test)
        
        # 计算指标
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        cv_scores = None
        if len(self.y_train) >= 10:
            cv_folds = self._get_cv_folds(max_cv=5)
            if cv_folds >= 2:
                cv_model = clone(self.model)
                # 交叉验证不会自动提供 eval_set，需关闭 early stopping
                try:
                    cv_model.set_params(early_stopping_rounds=None)
                except Exception:
                    pass
                cv_scores = cross_val_score(
                    cv_model, self.X_train, self.y_train,
                    cv=cv_folds, scoring='r2'
                )
        
        print("\n📈 模型评估结果:")
        print(f"{'指标':<20} {'训练集':<10} {'测试集':<10}")
        print(f"{'-'*50}")
        print(f"{'MAE (eV)':<20} {train_mae:.4f}      {test_mae:.4f}")
        print(f"{'RMSE (eV)':<20} {train_rmse:.4f}      {test_rmse:.4f}")
        print(f"{'R²分数':<20} {train_r2:.4f}      {test_r2:.4f}")
        if cv_scores is not None and np.any(np.isfinite(cv_scores)):
            print(f"\n交叉验证 R²分数: {np.nanmean(cv_scores):.4f} ± {np.nanstd(cv_scores):.4f}")
        else:
            print("\n训练样本较少，跳过交叉验证。")
        
        # 绘制预测 vs 实际
        self._plot_predictions(y_test_pred)
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_scores': cv_scores
        }
    
    def _plot_predictions(self, y_test_pred):
        """绘制预测结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 子图1: 预测vs实际
        ax1 = axes[0]
        ax1.scatter(self.y_test, y_test_pred, alpha=0.6, s=50)
        
        # 绘制完美预测线
        min_val = min(self.y_test.min(), y_test_pred.min())
        max_val = max(self.y_test.max(), y_test_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax1.set_xlabel('实际吸附能 (eV)', fontsize=12)
        ax1.set_ylabel('预测吸附能 (eV)', fontsize=12)
        ax1.set_title('预测 vs 实际 (测试集)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 残差图
        ax2 = axes[1]
        residuals = self.y_test - y_test_pred
        ax2.scatter(y_test_pred, residuals, alpha=0.6, s=50)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax2.set_xlabel('预测吸附能 (eV)', fontsize=12)
        ax2.set_ylabel('残差 (eV)', fontsize=12)
        ax2.set_title('残差分析', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        importance = self.model.feature_importances_
        feature_names = self.X_train.columns
        
        # 创建DataFrame
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 特征重要性 (Top {top_n}):")
        print(self.feature_importance.head(top_n).to_string(index=False))
        
        # 绘制特征重要性
        self._plot_feature_importance(top_n=top_n)
        
        return self.feature_importance
    
    def _plot_feature_importance(self, top_n=20):
        """绘制特征重要性图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 子图1: 条形图
        ax1 = axes[0]
        top_features = self.feature_importance.head(top_n)
        
        bars = ax1.barh(range(len(top_features)), top_features['importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.invert_yaxis()
        ax1.set_xlabel('特征重要性', fontsize=12)
        ax1.set_title(f'Top {top_n} 特征重要性', fontsize=14)
        
        # 在条形上添加数值
        for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
            ax1.text(val, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', ha='left', va='center')
        
        # 子图2: SHAP值分析
        ax2 = axes[1]
        if not SHAP_AVAILABLE:
            ax2.text(0.5, 0.5, '未安装 shap，跳过SHAP分析',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('SHAP值分析 (不可用)', fontsize=14)
        else:
            try:
                # 计算SHAP值并绘制均值绝对值条形图
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(self.X_test)
                shap_array = np.array(shap_values)
                if shap_array.ndim == 3:
                    shap_array = shap_array.mean(axis=0)
                mean_abs_shap = np.abs(shap_array).mean(axis=0)

                shap_df = pd.DataFrame({
                    'feature': self.X_test.columns,
                    'shap_value': mean_abs_shap
                }).sort_values('shap_value', ascending=False).head(top_n)

                ax2.barh(range(len(shap_df)), shap_df['shap_value'])
                ax2.set_yticks(range(len(shap_df)))
                ax2.set_yticklabels(shap_df['feature'])
                ax2.invert_yaxis()
                ax2.set_xlabel('mean(|SHAP value|)', fontsize=12)
                ax2.set_title('SHAP值分析', fontsize=14)
            except Exception as e:
                ax2.text(0.5, 0.5, f'SHAP分析失败:\n{e}',
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('SHAP值分析 (不可用)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_descriptor_relationships(self):
        """分析描述符与吸附能的关系"""
        print("\n🔬 描述符-吸附能关系分析...")

        if self.feature_importance is None:
            self._get_feature_importance()
        
        # 选择最重要的特征
        top_features = self.feature_importance.head(5)['feature'].tolist()
        
        if len(top_features) < 2:
            print("⚠️ 没有足够的特征进行分析")
            return
        
        # 创建关系图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        plotted_count = 0
        
        for feature in top_features[:6]:
            if feature in self.df.columns:
                x_data = self.df[feature]
            elif self.model_features is not None and feature in self.model_features.columns:
                x_data = self.model_features[feature]
            else:
                continue

            ax = axes[plotted_count]
            
            # 绘制散点图
            ax.scatter(x_data, self.target, alpha=0.6, s=50)
            
            # 添加趋势线
            if pd.Series(x_data).nunique() > 1:
                z = np.polyfit(x_data, self.target, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x_data)
                ax.plot(x_sorted, p(x_sorted), 'r--', lw=2)
            
            # 计算相关系数
            corr = np.corrcoef(x_data, self.target)[0, 1] if pd.Series(x_data).nunique() > 1 else np.nan
            corr_text = f"{corr:.3f}" if np.isfinite(corr) else "NaN"
            
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('吸附能 (eV)', fontsize=10)
            ax.set_title(f'{feature}\ncorr = {corr_text}', fontsize=11)
            ax.grid(True, alpha=0.3)
            plotted_count += 1

        if plotted_count == 0:
            print("⚠️ 没有可用于绘图的特征")
            return None
        
        # 隐藏多余的子图
        for j in range(plotted_count, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('关键描述符与吸附能的关系', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('descriptor_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出相关性分析
        print("\n📊 描述符相关性分析:")
        corr_df = self.model_features.copy()
        corr_df[self.target_column] = self.target
        correlation = corr_df.corr(numeric_only=True)[self.target_column].sort_values(
            key=abs, ascending=False
        )
        correlation = correlation.drop(self.target_column)
        
        print(correlation.head(10).to_string())
        
        return correlation
    
    def create_descriptor_rules(self, threshold=0.1):
        """从模型创建描述符规则"""
        print(f"\n📋 描述符规则提取 (重要性阈值: {threshold})...")
        
        if self.feature_importance is None:
            self._get_feature_importance()
        
        # 筛选重要特征
        important_features = self.feature_importance[
            self.feature_importance['importance'] > threshold
        ]
        
        if important_features.empty:
            print("⚠️ 没有特征超过重要性阈值")
            return
        
        rules = []
        
        for _, row in important_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # 获取特征统计数据
            if feature in self.df.columns:
                feature_series = self.df[feature]
                feature_kind = "原始特征"
            elif self.model_features is not None and feature in self.model_features.columns:
                feature_series = self.model_features[feature]
                feature_kind = "编码特征"
            else:
                continue

            mean_val = feature_series.mean()
            std_val = feature_series.std()
            if pd.Series(feature_series).nunique() > 1:
                corr = np.corrcoef(feature_series, self.target)[0, 1]
            else:
                corr = np.nan

            if np.isfinite(corr):
                direction = "正相关" if corr > 0 else "负相关"
                rule_text = f"{feature} {'增加' if corr > 0 else '减少'} → 吸附能{'增加' if corr > 0 else '减少'}"
                corr_text = f"{corr:.3f}"
            else:
                direction = "相关性不可用"
                rule_text = f"{feature} 变化对当前数据集相关性不稳定"
                corr_text = "NaN"

            rule = {
                '描述符': feature,
                '类型': feature_kind,
                '重要性': f"{importance:.4f}",
                '相关性': corr_text,
                '关系': direction,
                '平均值': f"{mean_val:.3f}",
                '标准差': f"{std_val:.3f}",
                '规则': rule_text
            }

            rules.append(rule)

        if not rules:
            print("⚠️ 未能从当前特征提取规则")
            return None
        
        # 创建规则DataFrame
        rules_df = pd.DataFrame(rules)
        
        print("\n📋 提取的描述符规则:")
        print(rules_df.to_string(index=False))
        
        # 保存规则
        rules_df.to_csv('descriptor_rules.csv', index=False, encoding='utf-8-sig')
        print("✅ 规则已保存: descriptor_rules.csv")
        
        return rules_df
    
    def predict_new_surfaces(self, new_data):
        """预测新表面的吸附能"""
        print("\n🔮 新表面预测...")
        
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 预处理新数据
        if isinstance(new_data, pd.DataFrame):
            new_data_processed = new_data.copy()
            
            # 处理分类变量
            if self.categorical_features:
                new_data_processed = pd.get_dummies(
                    new_data_processed, 
                    columns=[c for c in self.categorical_features if c in new_data_processed.columns],
                    drop_first=True
                )
            
            # 确保特征顺序一致
            missing_cols = set(self.X_train.columns) - set(new_data_processed.columns)
            extra_cols = set(new_data_processed.columns) - set(self.X_train.columns)
            
            for col in missing_cols:
                new_data_processed[col] = 0
            
            for col in extra_cols:
                new_data_processed = new_data_processed.drop(columns=[col])
            
            new_data_processed = new_data_processed[self.X_train.columns]
            
            # 标准化
            new_data_scaled = self.scaler.transform(new_data_processed)
            
            # 预测
            predictions = self.model.predict(new_data_scaled)
            
            # 创建结果DataFrame
            results = new_data.copy()
            results['预测吸附能(eV)'] = predictions
            results['置信度'] = 1.0  # 这里可以添加不确定性估计
            
            print("预测结果:")
            print(results.to_string(index=False))
            
            return results
        
        else:
            print("⚠️ 新数据应为DataFrame格式")
            return None
    
    def save_model(self, model_path='water_adsorption_model.json'):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有训练好的模型可保存")
        
        # 保存模型
        self.model.save_model(model_path)
        
        # 保存特征信息
        model_info = {
            'features': self.features.columns.tolist(),
            'target_name': self.target.name,
            'feature_importance': self.feature_importance.to_dict(),
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
        }
        
        import json
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ 模型已保存: {model_path}")
        print(f"✅ 模型信息已保存: model_info.json")
        
        return model_path
    
    def load_model(self, model_path='water_adsorption_model.json'):
        """加载模型"""
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        import json
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        print(f"✅ 模型已加载: {model_path}")
        return self.model


# 主程序
def main():
    """主程序"""
    print("="*60)
    print("水分子吸附能描述符分析系统")
    print("使用XGBoost发现表面性质与吸附能的关系")
    print("="*60)
    
    # 1. 加载数据
    # 替换为您的数据文件路径
    data_file = input("请输入数据文件路径 (CSV/Excel)，或按回车使用示例数据: ").strip()

    if data_file:
        if Path(data_file).exists():
            analyzer = WaterAdsorptionAnalyzer(data_file)
        else:
            print(f"⚠️ 文件不存在: {data_file}，将使用示例数据")
            analyzer = WaterAdsorptionAnalyzer()
    else:
        print("使用示例数据...")
        analyzer = WaterAdsorptionAnalyzer()

    df = analyzer.load_and_prepare_data(target_column='adsorption_energy')
    
    # 3. 数据预处理
    X_train, X_test, y_train, y_test = analyzer.preprocess_data(test_size=0.2)
    
    # 4. 训练模型
    use_grid_search = input("\n是否使用网格搜索优化参数? (y/N): ").strip().lower() == 'y'
    model = analyzer.train_xgboost_model(use_grid_search=use_grid_search)
    
    # 5. 分析描述符关系
    analyzer.analyze_descriptor_relationships()
    
    # 6. 创建描述符规则
    analyzer.create_descriptor_rules(threshold=0.01)
    
    # 7. 保存模型
    save_model = input("\n是否保存模型? (Y/n): ").strip().lower() != 'n'
    if save_model:
        analyzer.save_model()
    
    # 8. 新表面预测示例
    predict_new = input("\n是否预测新表面? (y/N): ").strip().lower() == 'y'
    if predict_new:
        # 创建示例新数据
        new_surface_data = pd.DataFrame([{
            'surface_element': 'Li',
            'atomic_radius': 1.52,
            'electronegativity': 0.98,
            'ionization_energy': 5.39,
            'electron_affinity': 0.62,
            'valence_electrons': 1,
            'surface_energy': 0.5,
            'work_function': 2.9,
            'band_gap': 2.5,
            'd_band_center': -2.0,
            'coordination_number': 4,
            'lattice_constant': 4.0,
            'surface_area': 25.0,
            'water_distance': 2.0,
            'water_angle': 104.5,
        }])
        
        predictions = analyzer.predict_new_surfaces(new_surface_data)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("生成的文件:")
    print("  - predictions_analysis.png      预测分析图")
    print("  - feature_importance_analysis.png 特征重要性图")
    print("  - descriptor_relationships.png   描述符关系图")
    print("  - descriptor_rules.csv          描述符规则")
    print("  - water_adsorption_model.json   训练好的模型")
    print("  - model_info.json              模型信息")
    print("="*60)


# 快速使用函数
def quick_analysis(data_file, target_col='adsorption_energy'):
    """
    快速分析函数
    
    参数:
    data_file: 数据文件路径
    target_col: 目标列名
    """
    analyzer = WaterAdsorptionAnalyzer(data_file)
    analyzer.load_and_prepare_data(target_column=target_col)
    analyzer.preprocess_data()
    analyzer.train_xgboost_model()
    analyzer.analyze_descriptor_relationships()
    analyzer.create_descriptor_rules()
    
    return analyzer


if __name__ == "__main__":
    # 安装必要包
    required_packages = ['xgboost', 'shap', 'seaborn', 'scikit-learn']
    print("需要的包: " + ", ".join(required_packages))
    print("安装命令: pip install " + " ".join(required_packages))
    
    # 运行主程序
    main()