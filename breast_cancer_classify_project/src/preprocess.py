"""
数据预处理模块
包括数据标准化、训练测试集划分等
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(X, y, test_size=0.3, random_state=42):
    """
    划分训练集和测试集
    
    Args:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标变量
        test_size (float): 测试集比例
        random_state (int): 随机种子
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"数据划分完成:")
    print(f"  训练集: {X_train.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    print(f"  训练集类别分布: 良性={sum(y_train==0)}, 恶性={sum(y_train==1)}")
    print(f"  测试集类别分布: 良性={sum(y_test==0)}, 恶性={sum(y_test==1)}")
    
    return X_train, X_test, y_train, y_test

def standardize_features(X_train, X_test):
    """
    标准化特征
    
    Args:
        X_train (np.ndarray): 训练集特征
        X_test (np.ndarray): 测试集特征
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("特征标准化完成")
    
    return X_train_scaled, X_test_scaled, scaler

def preprocess_data(X, y, test_size=0.3, standardize=True):
    """
    完整的预处理流程
    
    Args:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标变量
        test_size (float): 测试集比例
        standardize (bool): 是否标准化
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    
    # 标准化特征
    scaler = None
    if standardize:
        X_train, X_test, scaler = standardize_features(X_train, X_test)
    
    return X_train, X_test, y_train, y_test, scaler