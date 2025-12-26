"""
可视化模块
包括AdaBoost基分类器可视化、ROC曲线绘制等
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def plot_adaboost_tree_stump(tree_stump, feature_names, class_names, save_path=None):
    """
    绘制AdaBoost的决策树桩（基分类器）
    
    Args:
        tree_stump: 决策树桩模型
        feature_names (list): 特征名称列表
        class_names (list): 类别名称列表
        save_path (str): 图片保存路径
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制决策树桩（深度为1）
    plot_tree(tree_stump, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10,
              max_depth=1)
    
    plt.title('AdaBoost基分类器（决策树桩）', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        import os
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"AdaBoost决策树桩图已保存到: {save_path}")
    
    plt.show()

def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """
    绘制ROC曲线
    
    Args:
        fpr (np.ndarray): 假正例率
        tpr (np.ndarray): 真正例率
        auc_score (float): AUC值
        save_path (str): 图片保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('AdaBoost模型ROC曲线', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # 添加AUC值文本
    plt.text(0.6, 0.2, f'AUC = {auc_score:.4f}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线图已保存到: {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    绘制混淆矩阵热力图
    
    Args:
        cm (np.ndarray): 混淆矩阵
        class_names (list): 类别名称列表
        save_path (str): 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建热力图
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置坐标轴
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='AdaBoost模型混淆矩阵',
           ylabel='真实类别',
           xlabel='预测类别')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个单元格中添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存到: {save_path}")
    
    plt.show()

def plot_adaboost_convergence(model, X_test, y_test, save_path=None):
    """
    绘制AdaBoost模型随着弱分类器数量增加的性能收敛曲线
    
    Args:
        model: 训练好的AdaBoost模型
        X_test: 测试集特征
        y_test: 测试集标签
        save_path: 图片保存路径
    """
    # 获取模型在不同迭代阶段的预测
    n_estimators = len(model.estimators_)
    accuracy_scores = []
    
    # 逐步增加弱分类器数量，计算准确率
    for i in range(1, n_estimators + 1):
        # 使用前i个弱分类器进行预测
        y_pred = model.predict(X_test, n_estimators=i)
        accuracy = np.mean(y_pred == y_test)
        accuracy_scores.append(accuracy)
    
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_estimators + 1), accuracy_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('弱分类器数量', fontsize=12)
    plt.ylabel('测试集准确率', fontsize=12)
    plt.title('AdaBoost模型收敛曲线', fontsize=16)
    plt.grid(alpha=0.3)
    plt.ylim([0.8, 1.0])  # 根据实际情况调整
    
    # 标记最佳准确率
    best_acc = max(accuracy_scores)
    best_idx = accuracy_scores.index(best_acc)
    plt.plot(best_idx + 1, best_acc, 'ro', markersize=10)
    plt.annotate(f'最佳: {best_acc:.4f}', 
                 xy=(best_idx + 1, best_acc),
                 xytext=(best_idx + 5, best_acc - 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"收敛曲线图已保存到: {save_path}")
    
    plt.show()