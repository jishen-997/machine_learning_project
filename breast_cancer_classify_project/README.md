# 威斯康星乳腺癌分类项目

## 项目简介

本项目基于威斯康星乳腺癌数据集，使用决策树算法构建肿瘤良恶性分类模型。项目包含了数据预处理、特征分析、模型训练、可视化展示和结果评估全流程。

## 数据集

- **名称**：威斯康星乳腺癌数据集 (Wisconsin Breast Cancer Dataset)
- **来源**：UCI 机器学习库
- **样本数**：569 例
- **特征数**：30 个细胞核特征
- **目标变量**：诊断结果（良性/恶性）

## 项目结构

breast_cancer_project/
├── README.md # 项目说明
├── requirements.txt # 依赖包
├── data/ # 数据目录
│ └── wdbc.data # 数据集
| └── wdbc.names # 数据集说明
├── src/ # 核心代码
| ├── **init**.py # 初始化
| |── load_data.py # 数据加载
│ ├── preprocess.py # 预处理
│ ├── feature_analysis.py # 特征分析
│ ├── train_model.py # 模型训练
│ └── visualize.py # 可视化
| └── main.py # 主程序
├── models/ # 保存模型
│ └── decision_tree_model.pkl # 训练好的模型
└── results/ # 结果
| |── figures/ # 图片
│ ├── decision_tree.png
│ ├── feature_importance.png
│ └── roc_curve.png
└── report.pdf # 报告

text

## 安装与运行

### 环境要求

- Python 3.7+
- 安装依赖包：`pip install -r requirements.txt`

### 运行方式

1. 安装依赖：`pip install -r requirements.txt`
2. 下载数据集：将 wdbc.data 文件放在 data/目录下
3. 运行 Python 脚本：`python src/main.py`

## 主要功能

1. **数据加载与探索**：加载 WBCD 数据集并进行基本统计分析
2. **数据预处理**：数据标准化、训练测试集划分
3. **特征分析**：特征重要性评估和可视化
4. **模型训练**：决策树分类器训练
5. **模型评估**：准确率、精确率、召回率、F1 分数等指标
6. **可视化展示**：决策树结构、特征重要性、ROC 曲线等

## 模型性能

基础决策树模型在测试集上的表现：

- 准确率：约 94%
- 精确率：约 96%
- 召回率：约 92%
- F1 分数：约 94%

## 文件说明

-
- `src/`目录：模块化的 Python 代码，便于复用和扩展
- `models/`目录：保存训练好的模型
- `results/`目录：生成的图表和报告

## 作者

[吴极/20233001585]

## 参考

- UCI 机器学习库：https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
- Scikit-learn：https://scikit-learn.org/stable/
