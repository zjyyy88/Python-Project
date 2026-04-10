# xgboost_adsorption_descriptors 详细说明

## 1. 脚本做了什么

这个脚本的目标是：
- 用元素标签（例如 Co、Fe3+）和 d-band 信息构造可学习特征。
- 用 XGBoost 回归模型预测吸附能。
- 用留一交叉验证（LOOCV）评估小样本数据集上的泛化能力。
- 用 SHAP（若可用）或置换重要性解释“哪些描述符最重要”。

核心流程是：
1. 读取输入 CSV。
2. 解析元素标签并补充元素物理描述符。
3. 构建特征矩阵 X 和目标 y。
4. 运行 LOOCV 得到逐样本预测与评估指标。
5. 在全量数据上训练最终模型并做特征重要性。
6. 输出表格、图像和 summary 文本。

---

## 2. 输入数据要求

默认参数下要求输入列：
- Element：元素标签（支持 Co、Fe3+、Ni2+、O2- 这类格式）。
- d-band：d 带中心（数值）。
- Eads：吸附能目标（数值）。

可通过命令行参数修改列名：
- --element-col
- --dband-col
- --target-col

数据预处理规则：
- d-band 和目标列会用 to_numeric 转为数值，不可转换会变成 NaN。
- 缺失值处理：特征列用各列中位数填充（尤其是部分元素可能缺 vdw radius）。
- 会丢弃仍有 NaN 的样本行。
- 若清洗后样本数 < 8，会直接报错（因为结果非常不稳定）。

---

## 3. 特征工程与输出字段说明

### 3.1 构造的主要描述符

脚本会把 Element 转换成这些描述符：
- atomic_number：原子序数。
- atomic_mass：原子质量。
- covalent_radius：共价半径。
- vdw_radius：范德华半径。
- oxidation_state：氧化态（由标签解析，如 Fe3+ -> +3）。
- is_transition_metal：是否过渡金属（0/1）。
- is_lanthanide：是否镧系（0/1）。
- is_actinide：是否锕系（0/1）。
- d_band：从输入 d-band 列标准化命名得到。

可选参数：
- --use-element-onehot
  打开后会额外加入 one-hot 元素身份列（例如 el_Co、el_Cu）。

### 3.2 prepared_data.csv 含义

该文件保留了：
- 原始数据列。
- element_symbol（从标签提取的元素符号）。
- 衍生元素描述符列。
- d_band（用于模型训练的统一字段名）。

用途：
- 检查标签解析是否正确。
- 检查是否有异常值或缺失值填充后偏差。

### 3.3 loocv_predictions.csv 含义

关键列：
- y_true：真实吸附能。
- y_pred_loocv：该样本在“被留出”时的预测值。
- abs_error：绝对误差，计算为 |y_true - y_pred_loocv|。

判读建议：
- 按 abs_error 从大到小排序，优先排查误差最大的样本。
- 如果误差主要集中在特定元素或氧化态，说明模型对该化学环境学习不足。

### 3.4 descriptor_importance.csv 含义

列说明：
- feature：特征名。
- gain_importance：XGBoost 内部增益重要性（树分裂贡献）。
- perm_importance：置换重要性（打乱该特征后性能下降幅度）。
- shap_mean_abs：SHAP 平均绝对值（全局贡献强度）。

一般优先级：
- 如果 shap_mean_abs 可用，优先用它排序解释。
- 若 SHAP 不可用，用 perm_importance 解释。
- gain_importance 可以作为模型内部参考，但不应单独作为唯一结论。

---

## 4. 模型训练参数（非常详细）

脚本中的 XGBoost 参数如下：
- objective = reg:squarederror
- n_estimators = 300
- max_depth = 3
- learning_rate = 0.05
- subsample = 0.9
- colsample_bytree = 0.9
- reg_lambda = 2.0
- reg_alpha = 0.1
- min_child_weight = 1
- random_state = 42
- n_jobs = -1

下面逐项解释其含义与影响：

1) objective = reg:squarederror
- 含义：回归任务，优化平方误差。
- 影响：这是最常见且稳定的连续值回归目标。

2) n_estimators = 300
- 含义：提升树总棵树数量。
- 增大后：拟合能力更强，但过拟合风险和训练时间都会增加。
- 减小后：模型更简单，可能欠拟合。

3) max_depth = 3
- 含义：每棵树最大深度。
- 增大后：能学更复杂交互，但小数据上容易过拟合。
- 当前设置 3：偏保守，适合你这种样本量不大的数据。

4) learning_rate = 0.05
- 含义：每轮提升步长。
- 更小：学习更稳健，需要更多树。
- 更大：收敛更快但不稳定，易过拟合。

5) subsample = 0.9
- 含义：每棵树训练时随机采样样本比例。
- 作用：引入随机性，减轻过拟合。

6) colsample_bytree = 0.9
- 含义：每棵树可见的特征比例。
- 作用：减少特征共线性带来的过拟合，提升泛化。

7) reg_lambda = 2.0 (L2 正则)
- 含义：权重平方惩罚强度。
- 增大：模型更平滑，过拟合更少。
- 减小：模型更自由，可能过拟合。

8) reg_alpha = 0.1 (L1 正则)
- 含义：权重绝对值惩罚。
- 作用：压缩弱特征影响，提升稀疏性。

9) min_child_weight = 1
- 含义：子节点最小样本权重和。
- 增大：树更难继续分裂，更保守。
- 减小：树更容易分裂，模型更灵活。

10) random_state = 42
- 含义：随机种子，保证可复现性。

11) n_jobs = -1
- 含义：使用全部 CPU 核心并行训练。

---

## 5. 验证方式与指标解释

### 5.1 LOOCV（留一交叉验证）

机制：
- 总样本数为 N 时，会训练 N 次。
- 每次拿 1 个样本做测试，剩余 N-1 个训练。
- 最终每个样本都有 1 次“未见过该样本”的预测。

优点：
- 适合小样本。
- 最大化利用训练数据。

缺点：
- 计算成本比普通 K 折高。
- 结果对异常点敏感。

### 5.2 终端输出指标含义

脚本结束时会打印：
- Rows used：实际参与训练的样本数（清洗后）。
- Features used：实际模型特征数。
- LOOCV MAE：平均绝对误差，单位 eV，越小越好。
- LOOCV RMSE：均方根误差，单位 eV，对大误差更敏感，越小越好。
- LOOCV R2：决定系数。

R2 的常用判读：
- 接近 1：拟合较好。
- 约 0：与“直接预测均值”差不多。
- 小于 0：比均值基线还差，说明模型结构或特征需要改进。

---

## 6. 图像含义（重点）

### 6.1 loocv_parity.png

图的元素：
- x 轴：真实吸附能（True adsorption energy）。
- y 轴：LOOCV 预测吸附能（Predicted adsorption energy）。
- 红色虚线 y=x：理想预测线。

如何看：
- 点越贴近红线，说明该样本预测越准确。
- 若整体偏在线上方：模型倾向于预测偏高。
- 若整体偏在线下方：模型倾向于预测偏低。
- 若离散很大：模型泛化差，可能特征信息不足或样本太少。

### 6.2 descriptor_importance.png

图的元素：
- 横轴：重要性数值。
- 纵轴：特征名。
- 条形越长，特征对预测贡献越大。

使用哪种重要性由脚本自动决定：
- 如果 SHAP 可用：使用 shap_mean_abs（更推荐）。
- 否则：使用 perm_importance（置换重要性）。

注意事项：
- 重要性高不等于因果关系，只表示“对当前模型预测有用”。
- 相关特征之间可能互相分担贡献，导致单特征重要性被稀释。

---

## 7. 你当前一次运行结果解读（基于 summary.txt）

当前结果显示：
- SHAP available: True
- MAE = 0.1221 eV
- RMSE = 0.1638 eV
- R2 = 0.6586

这说明：
- 在当前 19 行样本上，模型有一定解释力（R2 > 0.6）。
- 但误差仍在 0.1 eV 量级，适合做趋势筛选，不适合高精度定量替代第一性原理计算。

Top descriptors（前 5）是：
1. d_band
2. atomic_number
3. vdw_radius
4. covalent_radius
5. is_transition_metal

化学含义可理解为：
- d_band 在你这组数据中仍是主导信号。
- 原子序数和半径类描述符提供了元素化学环境的二级修正。
- 是否过渡金属也具有辅助区分能力。

---

## 8. 如何把结果用于“找规律”

建议执行顺序：
1. 先看 loocv_parity.png：确认模型是否具备基本可用性。
2. 看 summary 的 MAE/RMSE/R2：判断误差等级。
3. 看 descriptor_importance.csv：确定核心描述符。
4. 看 loocv_predictions.csv：找到误差最大的样本并回溯化学成因。

实操建议：
- 如果 R2 下降明显或为负：
  - 增加样本量。
  - 检查标签一致性（元素、氧化态、单位）。
  - 试验是否加入 --use-element-onehot。
- 如果某些元素误差持续偏大：
  - 针对该元素补样本。
  - 引入更贴近化学环境的描述符（如电负性、配位环境特征等）。

---

## 9. 如果还想增加描述符，应该怎么操作（详细步骤）

下面给你一个可直接照做的流程，按顺序改就行。

### 9.1 先确定“描述符来源”

你可以把新描述符分成三类：
1. 元素固有属性：如 Pauling 电负性、第一电离能、原子体积。
2. 体系结构属性：如配位数、局域键长均值、表面层间距。
3. 组合/非线性属性：如 d_band × electronegativity、半径比等。

建议优先加“有物理意义”的描述符，再考虑复杂组合特征。

### 9.2 修改位置 1：在元素描述符函数里新增字段

文件函数位置：
- build_element_descriptors

你要做的事：
1. 计算新特征值。
2. 把它加入返回字典。

示例（以 electronegativity 为例）：

```python
def build_element_descriptors(symbol: str, oxidation_state: int) -> dict[str, float]:
  ...
  electronegativity = ...  # 你自己的数据来源
  return {
    ...
    "electronegativity": electronegativity,
  }
```

说明：
- 如果某些元素没有该值，先填 np.nan，后面会走中位数填补逻辑。

### 9.3 修改位置 2：把新字段加入建模特征列表

文件函数位置：
- prepare_dataset 里的 feature_cols

你要做的事：
1. 在 feature_cols 增加新字段名。
2. 确保字段名与上一步返回字典一致。

示例：

```python
feature_cols = [
  "d_band",
  "atomic_number",
  ...,
  "electronegativity",
]
```

### 9.4 修改位置 3：若是“行级结构描述符”，在 prepare_dataset 里计算

如果描述符不是元素固有属性，而是每一行样本独有（例如你提前算好的配位数），做法是：
1. 确保输入 CSV 有这一列，比如 coordination_number。
2. 在 prepare_dataset 中对这列做数值化。
3. 把它加到 feature_cols。

示例：

```python
out_df["coordination_number"] = pd.to_numeric(out_df["coordination_number"], errors="coerce")
feature_cols = [..., "coordination_number"]
```

### 9.5 修改后如何验证“描述符真的生效”

每次新增描述符后，建议按下面 4 步检查：
1. 看 prepared_data.csv 是否出现新列，且数值合理。
2. 看终端输出 Features used 是否增加。
3. 看 descriptor_importance.csv 是否出现该特征。
4. 对比新增前后的 MAE、RMSE、R2 是否改进。

### 9.6 如何判断“该描述符有价值”

可用以下标准：
1. 在 LOOCV 下，R2 提升且 MAE/RMSE 下降。
2. 在 descriptor_importance.csv 中有稳定非零贡献。
3. 在化学上可解释，不只是偶然拟合。

### 9.7 常见坑与规避

1. 列名拼写不一致
- 症状：报 KeyError 或新特征未进入模型。
- 规避：统一使用同一个字符串，并在 prepared_data.csv 检查。

2. 新特征全是常数
- 症状：重要性接近 0。
- 规避：先检查该列的唯一值数量。

3. 缺失值过多
- 症状：模型性能波动大。
- 规避：优先提升原始数据质量，不要完全依赖中位数填补。

4. 一次加太多特征
- 症状：结果变好但不稳定，解释困难。
- 规避：每轮只加 1 到 2 个描述符，逐步对比。

### 9.8 推荐的增量实验模板

建议你按下面节奏做：
1. 基线：当前 9 个特征，记录 MAE/RMSE/R2。
2. +1 个候选描述符：重新运行并记录变化。
3. 若提升明显，保留；若无提升或退化，回滚。
4. 继续下一个候选描述符。

这样可以快速找到“真正有用”的描述符，而不是盲目堆特征。

---

## 10. 输出文件总表

输出目录默认是：
- 脚本同目录下的 xgboost_descriptor_results

文件说明：
- prepared_data.csv：清洗和特征扩展后的数据。
- loocv_predictions.csv：逐样本 LOOCV 预测与误差。
- descriptor_importance.csv：特征重要性明细。
- loocv_parity.png：真实值-预测值一致性图。
- descriptor_importance.png：特征重要性柱状图。
- summary.txt：关键指标和前 5 描述符摘要。

以上文件组合在一起，既能做模型质量判断，也能支撑后续的化学规律归纳。
