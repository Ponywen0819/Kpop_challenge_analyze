import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse

def calculate_top_k_metrics(df, k=10):
    """
    计算 top-k 的评估指标
    
    参数:
    df: DataFrame，包含预测结果
    k: int，top-k 的值，默认为 10
    
    返回:
    tuple: (准确率, MRR)
    """
    y_pred = []  # 用于存储准确率
    reciprocal_ranks = []  # 用于存储 MRR

    # 对每一行数据进行评估
    for _, row in df.iterrows():
        # 获取实际合作者
        actual = row['collaborator']
        
        # 获取预测的 top-k 推荐列表
        predicted = row['recommend_idol'].split(', ')[:k]
        
        # 计算准确率
        if actual in predicted:
            y_pred.append(1)
        else:
            y_pred.append(0)
        
        # 计算 MRR
        try:
            rank = predicted.index(actual) + 1  # 找到第一个正确答案的位置
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)  # 如果没有找到正确答案，MRR 为 0
    
    accuracy = np.mean(y_pred)
    mrr = np.mean(reciprocal_ranks)
    
    return accuracy, mrr

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='计算推荐系统的 top-k 评估指标')
    parser.add_argument('--file', type=str, default='predictions.csv',
                      help='要评估的 CSV 文件路径 (默认: predictions.csv)')
    parser.add_argument('--k', type=int, default=10,
                      help='top-k 的值 (默认: 10)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 读取数据
        df = pd.read_csv(args.file)
        
        # 计算 top-k 指标
        accuracy, mrr = calculate_top_k_metrics(df, k=args.k)
        
        # 打印结果
        print(f"Top-{args.k} 评估指标:")
        print(f"Top-{args.k} 准确率 (Top-{args.k} Accuracy): {accuracy:.4f}")
        print(f"MRR (MRR): {mrr:.4f}")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{args.file}'")
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == "__main__":
    main()
