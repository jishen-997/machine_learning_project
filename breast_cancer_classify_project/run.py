"""
运行脚本 - 从项目根目录运行
"""
import sys
import os

# 添加src目录到Python路径
sys.path.append('src')

from src.main import main

if __name__ == "__main__":
    print("="*70)
    print("乳腺癌诊断系统 - AdaBoost算法")
    print("="*70)
    print("项目目录:", os.getcwd())
    print("="*70)
    
    main()