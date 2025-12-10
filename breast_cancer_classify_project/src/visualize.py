"""
可视化模块
包括决策树可视化、ROC曲线绘制等
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import graphviz
from sklearn.tree import export_graphviz

def plot_decision_tree(model, feature_names, class_names, max_depth=4, save_path=None):
    """
    绘制决策树结构图
    
    Args:
        model: 决策树模型
        feature_names (list): 特征名称列表
        class_names (list): 类别名称列表
        max_depth (int): 显示的最大深度
        save_path (str): 图片保存路径
    """
    plt.figure(figsize=(20, 12))
    
    # 绘制决策树
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10,
              max_depth=max_depth)
    
    plt.title(f'决策树分类器 (最大深度={max_depth})', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"决策树结构图已保存到: {save_path}")
    
    plt.show()

def export_decision_tree_graphviz(model, feature_names, class_names, save_path):
    """
    导出决策树为Graphviz格式（可生成更美观的树图）
    
    Args:
        model: 决策树模型
        feature_names (list): 特征名称列表
        class_names (list): 类别名称列表
        save_path (str): 文件保存路径（不包含扩展名）
    """
    # 导出为dot格式
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    # 保存为dot文件
    dot_file = save_path + '.dot'
    with open(dot_file, 'w') as f:
        f.write(dot_data)
    
    print(f"决策树Graphviz文件已保存到: {dot_file}")
    print("可以使用以下命令转换为PNG: dot -Tpng {} -o {}.png".format(dot_file, save_path))

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
    plt.title('ROC曲线', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # 添加AUC值文本
    plt.text(0.6, 0.2, f'AUC = {auc_score:.4f}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if save_path:
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
           title='混淆矩阵',
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵图已保存到: {save_path}")
    
    plt.show()

def plot_feature_distributions(X, y, feature_names, target_names, top_n=5, save_path=None):
    """
    绘制重要特征的分布图
    
    Args:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标变量
        feature_names (list): 特征名称列表
        target_names (list): 目标类别名称列表
        top_n (int): 显示前N个特征
        save_path (str): 图片保存路径（前缀）
    """
    import pandas as pd
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['diagnosis'] = y
    
    # 计算特征与目标的相关性
    correlations = []
    for feature in feature_names:
        corr = np.corrcoef(df[feature], y)[0, 1]
        correlations.append((feature, abs(corr)))
    
    # 按相关性排序
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [feat for feat, _ in correlations[:top_n]]
    
    # 绘制分布图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features[:6]):  # 最多显示6个特征
        # 良性样本
        benign_data = df[df['diagnosis'] == 0][feature]
        # 恶性样本
        malignant_data = df[df['diagnosis'] == 1][feature]
        
        axes[i].hist(benign_data, alpha=0.5, label=target_names[0], bins=30, color='skyblue')
        axes[i].hist(malignant_data, alpha=0.5, label=target_names[1], bins=30, color='salmon')
        
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('频数')
        axes[i].set_title(f'{feature}分布')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    # 如果特征少于6个，隐藏多余的子图
    for i in range(len(top_features[:6]), 6):
        fig.delaxes(axes[i])
    
    plt.suptitle('Top特征分布对比 (按与目标的相关性排序)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        full_path = f"{save_path}_feature_distributions.png"
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存到: {full_path}")
    
    plt.show()