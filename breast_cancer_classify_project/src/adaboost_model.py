"""
AdaBoost模型模块
实现AdaBoost算法用于乳腺癌分类
"""
import os
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

class AdaBoostModel:
    """
    AdaBoost分类器封装类
    使用决策树桩作为基分类器
    """
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=42):
        """
        初始化AdaBoost模型
        
        Args:
            n_estimators (int): 弱分类器的数量
            learning_rate (float): 学习率，控制每个弱分类器的贡献
            random_state (int): 随机种子
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # 创建基分类器（决策树桩，深度为1）
        base_estimator = DecisionTreeClassifier(
            max_depth=1,
            random_state=random_state
        )
        
        # 初始化AdaBoost模型
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state
        )
        
        self.feature_importance = None
        self.training_time = None
        self.evaluation_results = {}
        
    def train(self, X_train, y_train):
        """
        训练AdaBoost模型
        
        Args:
            X_train (np.ndarray): 训练集特征
            y_train (np.ndarray): 训练集标签
        """
        start_time = time.time()
        
        print(f"开始训练AdaBoost模型...")
        print(f"模型参数: n_estimators={self.n_estimators}, learning_rate={self.learning_rate}")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        print(f"模型训练完成，耗时: {self.training_time:.2f}秒")
        
        # 计算训练集性能
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        self.evaluation_results['train_accuracy'] = train_accuracy
        print(f"训练集准确率: {train_accuracy:.4f}")
        
        # 获取特征重要性
        self.feature_importance = self.model.feature_importances_
        
        # 显示弱分类器数量
        print(f"弱分类器数量: {len(self.model.estimators_)}")
        print(f"弱分类器权重: {self.model.estimator_weights_[:5]}...")  # 只显示前5个
        
    def evaluate(self, X_test, y_test, target_names=None):
        """
        评估模型性能
        
        Args:
            X_test (np.ndarray): 测试集特征
            y_test (np.ndarray): 测试集标签
            target_names (list): 目标类别名称列表
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        print("\n评估AdaBoost模型性能...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 混淆矩阵
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
        
        self.evaluation_results.update(results)
        
        # 打印评估结果
        print(f"测试集评估结果:")
        print(f"  准确率 (Accuracy): {accuracy:.4f}")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  召回率 (Recall): {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  AUC: {roc_auc:.4f}")
        
        # 打印分类报告
        if target_names is None:
            target_names = ['良性', '恶性']
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        # 打印混淆矩阵
        print("混淆矩阵:")
        print(f"         预测良性  预测恶性")
        print(f"实际良性    {cm[0,0]:4d}       {cm[0,1]:4d}")
        print(f"实际恶性    {cm[1,0]:4d}       {cm[1,1]:4d}")
        
        return results
    
    def get_feature_importance_df(self, feature_names):
        """
        获取特征重要性DataFrame
        
        Args:
            feature_names (list): 特征名称列表
            
        Returns:
            pd.DataFrame: 特征重要性DataFrame
        """
        if self.feature_importance is None:
            return None
            
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def save_model(self, filepath):
        """
        保存模型到文件
        
        Args:
            filepath (str): 模型保存路径
        """
        model_dir = os.path.dirname(filepath)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        joblib.dump(self.model, filepath)
        print(f"AdaBoost模型已保存到: {os.path.abspath(filepath)}")
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        Args:
            filepath (str): 模型文件路径
        """
        self.model = joblib.load(filepath)
        print(f"已从 {filepath} 加载AdaBoost模型")
        
        # 重新计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

def train_adaboost(X_train, X_test, y_train, y_test, 
                  n_estimators=50, learning_rate=1.0, 
                  model_save_path='../models/adaboost_model.pkl'):
    """
    完整的AdaBoost训练流程
    
    Args:
        X_train (np.ndarray): 训练集特征
        X_test (np.ndarray): 测试集特征
        y_train (np.ndarray): 训练集标签
        y_test (np.ndarray): 测试集标签
        n_estimators (int): 弱分类器的数量
        learning_rate (float): 学习率
        model_save_path (str): 模型保存路径
        
    Returns:
        tuple: (model, results)
            model: 训练好的AdaBoostModel实例
            results: 评估结果字典
    """
    # 初始化模型
    adaboost_model = AdaBoostModel(
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    
    # 训练模型
    adaboost_model.train(X_train, y_train)
    
    # 评估模型
    results = adaboost_model.evaluate(X_test, y_test)
    
    # 保存模型
    if model_save_path:
        adaboost_model.save_model(model_save_path)
    
    return adaboost_model, results


def explain_adaboost_algorithm():
    """
    解释AdaBoost算法原理
    """
    print("\n" + "="*70)
    print("AdaBoost算法原理")
    print("="*70)
    print("""
    AdaBoost（自适应增强）算法原理:
    
    1. 初始化权重: 为每个训练样本分配相同的权重
    2. 迭代训练弱分类器:
       a. 使用当前样本权重训练一个弱分类器（通常是决策树桩）
       b. 计算该弱分类器的错误率
       c. 根据错误率计算该弱分类器的权重（错误率越低，权重越高）
       d. 更新样本权重：增加分类错误样本的权重，减少分类正确样本的权重
    3. 组合弱分类器: 将所有弱分类器的预测结果加权投票得到最终预测
    
    关键特点:
    - 自适应调整样本权重，关注难以分类的样本
    - 每个弱分类器只关注前一个分类器分错的样本
    - 最终模型是多个弱分类器的加权组合
    - 对异常值敏感，但通常比单一分类器表现更好
    
    在本项目中:
    - 使用决策树桩（最大深度为1的决策树）作为弱分类器
    - 设置了{n_estimators}个弱分类器
    - 学习率为{learning_rate}，控制每个弱分类器的贡献程度
    """)