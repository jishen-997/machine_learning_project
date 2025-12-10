from load_data import load_data
from preprocess import preprocess_data
from feature_analysis import calculate_feature_importance, plot_feature_importance
from train_model import train_decision_tree
from visualize import plot_decision_tree, plot_roc_curve

def main():
    # 加载数据
    X, y, feature_names, target_names = load_data()
    
    # 预处理
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
    
    # 特征分析
    feature_importance_df = calculate_feature_importance(X_train, y_train, feature_names)
    plot_feature_importance(feature_importance_df, save_path='../results/figures/feature_importance.png')
    
    # 训练模型
    dt_model, results = train_decision_tree(X_train, X_test, y_train, y_test)
    
    # 可视化
    plot_decision_tree(dt_model.model, feature_names, target_names, save_path='../results/figures/decision_tree.png')
    plot_roc_curve(results['roc_curve'][0], results['roc_curve'][1], results['auc'], 
                  save_path='../results/figures/roc_curve.png')

if __name__ == "__main__":
    main()