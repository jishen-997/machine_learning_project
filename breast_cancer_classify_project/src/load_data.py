"""
数据加载模块
支持从本地文件加载和从sklearn加载两种方式
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import os

def load_data_from_file(file_path):
    """
    从本地文件加载威斯康星乳腺癌数据集
    
    Args:
        file_path (str): 数据文件路径
        
    Returns:
        tuple: (X, y, feature_names, target_names)
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (n_samples,)
            feature_names: 特征名称列表
            target_names: 目标类别名称列表
    """
    try:
        # 列名定义
        column_names = ['id', 'diagnosis']
        
        # 添加30个特征名称（基于原始数据集描述）
        feature_mean = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                       'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
        
        feature_se = ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 
                     'smoothness_se', 'compactness_se', 'concavity_se', 
                     'concave points_se', 'symmetry_se', 'fractal_dimension_se']
        
        feature_worst = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                        'smoothness_worst', 'compactness_worst', 'concavity_worst',
                        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        
        feature_names = feature_mean + feature_se + feature_worst
        column_names.extend(feature_names)
        
        # 加载数据
        data = pd.read_csv(file_path, header=None, names=column_names)
        
        # 分离特征和目标变量
        X = data[feature_names].values
        y = data['diagnosis'].values
        
        # 将目标变量转换为数值（M=恶性=1, B=良性=0）
        y = np.array([1 if label == 'M' else 0 for label in y])
        
        # 目标类别名称
        target_names = ['良性 (B)', '恶性 (M)']
        
        print(f"数据加载成功: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
        print(f"类别分布: 良性={sum(y==0)}, 恶性={sum(y==1)}")
        
        return X, y, feature_names, target_names
        
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，使用sklearn内置数据集")
        return load_data_from_sklearn()

def load_data_from_sklearn():
    """
    从sklearn加载威斯康星乳腺癌数据集
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = ['良性 (B)', '恶性 (M)']  # 自定义名称以保持一致性
    
    print(f"从sklearn加载数据成功: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    print(f"类别分布: {target_names[0]}={sum(y==0)}, {target_names[1]}={sum(y==1)}")
    
    return X, y, feature_names, target_names

def load_data():
    """
    主加载函数，尝试从文件加载，失败则从sklearn加载
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    # 尝试从本地文件加载
    file_path = '../data/wdbc.data'
    
    try:
        # 检查文件是否存在
        if os.path.exists(file_path):
            return load_data_from_file(file_path)
        else:
            print(f"文件 {file_path} 不存在，使用sklearn内置数据集")
            return load_data_from_sklearn()
    except Exception as e:
        print(f"从文件加载失败: {e}")
        print("使用sklearn内置数据集")
        return load_data_from_sklearn()