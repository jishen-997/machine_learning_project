####乳腺癌诊断系统 - AdaBoost 算法####
项目简介
使用 AdaBoost 集成学习算法对威斯康星乳腺癌数据集进行分类分析。

快速开始

1. 安装依赖
   bash
   pip install -r requirements.txt
2. 运行程序
   bash
   python run.py
   项目结构
   text
   BREAST_CANCER_CLASSIFY_PRO/
   ├── data/ # 数据集
   │ ├── wdbc.data # 原始数据文件
   │ └── wdbc.names # 数据描述文件
   ├── models/ # 模型保存
   │ └── adaboost_model.pkl # AdaBoost 模型
   ├── results/ # 结果输出
   │ └── figures/ # 可视化图表
   ├── src/ # 源代码
   │ ├── load_data.py # 数据加载
   │ ├── preprocess.py # 数据预处理
   │ ├── feature_analysis.py # 特征分析
   │ ├── adaboost_model.py # AdaBoost 模型
   │ ├── visualize.py # 可视化
   │ └── main.py # 主程序
   ├── run.py # 运行脚本
   ├── requirements.txt # 依赖包
   └── README.md # 项目说明
   算法说明
   AdaBoost 原理
   AdaBoost（自适应增强）通过组合多个弱分类器（决策树桩）构建强分类器：

初始化样本权重

迭代训练弱分类器（决策树桩）

根据错误率调整样本权重

加权组合弱分类器

参数设置
n_estimators=50：50 个决策树桩

learning_rate=1.0：学习率

base_estimator=DecisionTreeClassifier(max_depth=1)：基分类器为深度 1 的决策树

数据集
威斯康星乳腺癌诊断数据集

569 个样本（212 恶性，357 良性）

30 个特征（半径、纹理、周长等）

二分类任务（恶性/良性）

预期结果
运行后将生成：

控制台输出：模型训练过程和性能评估

可视化图表：

特征重要性图

特征相关性热力图

决策树桩可视化

ROC 曲线图

模型文件：models/adaboost_model.pkl

依赖包
text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
注意事项
确保 Python 版本为 3.8+

首次运行前需安装依赖包

如缺少数据文件，程序会自动从 scikit-learn 加载
