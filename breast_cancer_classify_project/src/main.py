"""
主程序入口 - 只使用AdaBoost算法
"""
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.dirname(__file__))

from load_data import load_data
from preprocess import preprocess_data
from feature_analysis import calculate_feature_importance, plot_feature_importance, analyze_correlations
from adaboost_model import train_adaboost
from visualize import plot_adaboost_tree_stump, plot_roc_curve

def main():
    # 加载数据
    print("="*70)
    print("威斯康星乳腺癌数据集 - AdaBoost算法分析")
    print("="*70)
    
    X, y, feature_names, target_names = load_data()
    
    # 预处理
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
    
    # 特征分析
    print("\n" + "="*70)
    print("特征重要性分析")
    print("="*70)
    feature_importance_df = calculate_feature_importance(X_train, y_train, feature_names, method='random_forest')
    plot_feature_importance(feature_importance_df, save_path='../results/figures/feature_importance.png')
    
    # 分析特征相关性
    analyze_correlations(X_train, feature_names, save_path='../results/figures/correlation_heatmap.png')
    
    # 训练AdaBoost模型
    print("\n" + "="*70)
    print("AdaBoost模型训练")
    print("="*70)
    
    # 训练AdaBoost模型
    adaboost_model, results = train_adaboost(
        X_train, X_test, 
        y_train, y_test,
        n_estimators=50,
        learning_rate=1.0
    )
    
    # 可视化AdaBoost的第一个基分类器（决策树桩）
    if hasattr(adaboost_model.model, 'estimators_') and len(adaboost_model.model.estimators_) > 0:
        print("\n" + "="*70)
        print("AdaBoost基分类器可视化")
        print("="*70)
        # 可视化第一个基分类器（深度为1的决策树桩）
        plot_adaboost_tree_stump(
            adaboost_model.model.estimators_[0], 
            feature_names, 
            target_names,
            save_path='../results/figures/adaboost_tree_stump.png'
        )
    
    # 绘制ROC曲线
    print("\n" + "="*70)
    print("模型性能评估")
    print("="*70)
    
    plot_roc_curve(
        results['roc_curve'][0], 
        results['roc_curve'][1], 
        results['auc'],
        save_path='../results/figures/roc_curve.png'
    )
    
    # 显示AdaBoost的特征重要性
    if adaboost_model.feature_importance is not None:
        print("\n" + "="*70)
        print("AdaBoost特征重要性")
        print("="*70)
        
        # 创建特征重要性DataFrame
        ada_importance_df = adaboost_model.get_feature_importance_df(feature_names)
        
        # 打印Top 10重要特征
        print("Top 10重要特征:")
        for i, row in ada_importance_df.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # 可视化AdaBoost的特征重要性
        plot_feature_importance(
            ada_importance_df, 
            top_n=15, 
            save_path='../results/figures/adaboost_feature_importance.png'
        )
    
    print("\n" + "="*70)
    print("AdaBoost算法分析完成!")
    print("="*70)
    
    # 显示最终模型性能
    print(f"\nAdaBoost模型最终性能:")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  精确率: {results['precision']:.4f}")
    print(f"  召回率: {results['recall']:.4f}")
    print(f"  F1分数: {results['f1_score']:.4f}")
    print(f"  AUC值: {results['auc']:.4f}")

if __name__ == "__main__":
    main()