"""
特征分析模块
包括特征重要性评估和可视化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def calculate_feature_importance(X_train, y_train, feature_names, method='random_forest'):
    """
    计算特征重要性
    
    Args:
        X_train (np.ndarray): 训练集特征
        y_train (np.ndarray): 训练集标签
        feature_names (list): 特征名称列表
        method (str): 计算方法，可选 'decision_tree' 或 'random_forest'
        
    Returns:
        pd.DataFrame: 特征重要性DataFrame，包含特征名和重要性分数
    """
    if method == 'decision_tree':
        # 使用决策树计算特征重要性
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        importance = model.feature_importances_
    else:
        # 使用随机森林计算特征重要性（更稳定）
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        importance = model.feature_importances_
    
    # 创建DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print(f"使用{method}计算特征重要性完成")
    print("Top 5重要特征:")
    for i, row in feature_importance_df.head().iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance_df

def plot_feature_importance(feature_importance_df, top_n=15, save_path=None):
    """
    可视化特征重要性
    
    Args:
        feature_importance_df (pd.DataFrame): 特征重要性DataFrame
        top_n (int): 显示前N个重要特征
        save_path (str): 图片保存路径
    """
    # 选择前N个特征
    top_features = feature_importance_df.head(top_n)
    
    # 设置图表样式
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    
    # 创建水平条形图
    bars = plt.barh(range(top_n), top_features['importance'].values, color=colors)
    plt.yticks(range(top_n), top_features['feature'].values)
    plt.gca().invert_yaxis()  # 重要性从高到低显示
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    plt.xlabel('特征重要性')
    plt.title(f'Top {top_n} 特征重要性排名')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图表已保存到: {save_path}")
    
    plt.show()

def analyze_correlations(X, feature_names, save_path=None):
    """
    分析特征间的相关性
    
    Args:
        X (np.ndarray): 特征矩阵
        feature_names (list): 特征名称列表
        save_path (str): 图片保存路径
    """
    # 创建DataFrame以便计算相关性
    df = pd.DataFrame(X, columns=feature_names)
    
    # 计算相关性矩阵
    corr_matrix = df.corr()
    
    # 可视化相关性矩阵
    plt.figure(figsize=(14, 12))
    
    # 创建热力图
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('特征相关性矩阵热力图', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相关性热力图已保存到: {save_path}")
    
    plt.show()
    
    # 找出高度相关的特征对（|相关性| > 0.9）
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print("\n高度相关的特征对 (|相关性| > 0.9):")
        for feature1, feature2, corr in high_corr_pairs:
            print(f"  {feature1} 与 {feature2}: {corr:.4f}")
    else:
        print("\n没有发现高度相关的特征对 (|相关性| > 0.9)")
    
    return corr_matrix, high_corr_pairs