"""
模型训练模块
包括决策树模型的训练、评估和保存
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

class DecisionTreeModel:
    """
    决策树分类器封装类
    """
    def __init__(self, max_depth=4, min_samples_split=2, 
                 min_samples_leaf=1, random_state=42):
        """
        初始化决策树模型
        
        Args:
            max_depth (int): 树的最大深度
            min_samples_split (int): 内部节点再划分所需最小样本数
            min_samples_leaf (int): 叶节点所需最小样本数
            random_state (int): 随机种子
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        # 初始化模型
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        
        self.feature_importance = None
        self.training_accuracy = None
        self.testing_accuracy = None
        
    def train(self, X_train, y_train):
        """
        训练决策树模型
        
        Args:
            X_train (np.ndarray): 训练集特征
            y_train (np.ndarray): 训练集标签
        """
        print("开始训练决策树模型...")
        print(f"模型参数: max_depth={self.max_depth}, "
              f"min_samples_split={self.min_samples_split}, "
              f"min_samples_leaf={self.min_samples_leaf}")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 计算训练集准确率
        y_train_pred = self.model.predict(X_train)
        self.training_accuracy = accuracy_score(y_train, y_train_pred)
        
        # 获取特征重要性
        self.feature_importance = self.model.feature_importances_
        
        print(f"模型训练完成")
        print(f"训练集准确率: {self.training_accuracy:.4f}")
        
    def evaluate(self, X_test, y_test, feature_names=None, target_names=None):
        """
        评估模型性能
        
        Args:
            X_test (np.ndarray): 测试集特征
            y_test (np.ndarray): 测试集标签
            feature_names (list): 特征名称列表
            target_names (list): 目标类别名称列表
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        print("\n开始评估模型性能...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 保存测试集准确率
        self.testing_accuracy = accuracy
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 计算ROC曲线和AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # 创建评估结果字典
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr),
            'auc': roc_auc,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # 打印评估结果
        print(f"测试集评估结果:")
        print(f"  准确率 (Accuracy): {accuracy:.4f}")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall): {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  AUC: {roc_auc:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        if target_names is None:
            target_names = ['良性', '恶性']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # 打印混淆矩阵
        print("混淆矩阵:")
        print(f"         预测良性  预测恶性")
        print(f"实际良性    {cm[0,0]:4d}       {cm[0,1]:4d}")
        print(f"实际恶性    {cm[1,0]:4d}       {cm[1,1]:4d}")
        
        return results
    
    def save_model(self, filepath):
        """
        保存模型到文件
        
        Args:
            filepath (str): 模型保存路径
        """
        joblib.dump(self.model, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        Args:
            filepath (str): 模型文件路径
        """
        self.model = joblib.load(filepath)
        print(f"模型已从 {filepath} 加载")

def train_decision_tree(X_train, X_test, y_train, y_test, 
                        max_depth=4, min_samples_split=2, 
                        min_samples_leaf=1, model_save_path='../models/decision_tree_model.pkl'):
    """
    完整的决策树训练流程
    
    Args:
        X_train (np.ndarray): 训练集特征
        X_test (np.ndarray): 测试集特征
        y_train (np.ndarray): 训练集标签
        y_test (np.ndarray): 测试集标签
        max_depth (int): 树的最大深度
        min_samples_split (int): 内部节点再划分所需最小样本数
        min_samples_leaf (int): 叶节点所需最小样本数
        model_save_path (str): 模型保存路径
        
    Returns:
        tuple: (model, results)
            model: 训练好的DecisionTreeModel实例
            results: 评估结果字典
    """
    # 初始化模型
    dt_model = DecisionTreeModel(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    
    # 训练模型
    dt_model.train(X_train, y_train)
    
    # 评估模型
    results = dt_model.evaluate(X_test, y_test)
    
    # 保存模型
    if model_save_path:
        dt_model.save_model(model_save_path)
    
    return dt_model, results