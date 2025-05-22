import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime, timedelta
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from gnn_best.model import GNNRecommender
from gnn_best.util import prepare_graph_data, get_node_encoder
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


def evaluate_model(model, all_data, node_encoder, top_k=10):
    """
    評估模型在測試集上的表現，只使用歷史數據生成 embedding
    """
    model.eval()
    
    # 按時間順序處理測試數據
    test_df = all_data[all_data['timestamp'] >= datetime(2025, 1, 1).timestamp()]
    
    test_df = test_df.sort_values('timestamp')
    all_recommendations = []
    
    for _, row in test_df.iterrows():
        start_timestamp = row['timestamp'] - 60 * 60 * 24 * 60
        # 使用歷史數據生成 embedding
        with torch.no_grad():
            data= prepare_graph_data(all_data, node_encoder, start_timestamp=start_timestamp, end_timestamp=row['timestamp']).to('cuda')
            embeddings = model.get_embedding(data.x, data.edge_index, data.edge_attr)
        
        # 為當前時間點的藝人生成推薦
        
        artist = row['source']
        
        artist_idx = node_encoder.transform([artist])[0]
        artist_embedding = embeddings[artist_idx]
                
        similarities = F.cosine_similarity(artist_embedding.unsqueeze(0), embeddings).to('cpu')
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k+1)
                
        recommendations = []
        for idx in top_k_indices[1:]:
            rec_artist = node_encoder.inverse_transform([idx])[0]
            recommendations.append(rec_artist)
                
        all_recommendations.append({
            'source': artist,
            'label': row['target'],
            'recommendations': recommendations
        })
    
    # 計算評估指標
    true_positives = 0
    total_recommendations = 0
    reciprocal_ranks = []
    hits_at_k = 0
    
    for recommendations in all_recommendations:
        if recommendations['label'] in recommendations['recommendations']:
            true_positives += 1
            
            rank = recommendations['recommendations'].index(recommendations['label']) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank <= top_k:
                hits_at_k += 1
        total_recommendations += len(recommendations['recommendations'])
        
        
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    hit_at_k = hits_at_k / len(all_recommendations) if len(all_recommendations) > 0 else 0
    
    return {
        'mrr': mrr,
        f'hit@{top_k}': hit_at_k
    }

def generate_predictions(model, test_data, test_df, node_encoder, top_k=10):
    """
    生成預測結果並保存為 CSV 文件
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embedding(test_data.x, test_data.edge_index, test_data.edge_attr)
    
    # 創建預測結果列表
    predictions = []
    
    # 對每個藝人生成推薦
    for artist in test_df['source'].unique():
        artist_idx = node_encoder.transform([artist])[0]
        artist_embedding = embeddings[artist_idx]
        
        # 計算與其他藝人的相似度
        similarities = F.cosine_similarity(artist_embedding.unsqueeze(0), embeddings)
        
        # 獲取 top-k 推薦
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k+1)
        
        # 轉換回藝人名字
        recommendations = []
        for i,idx in enumerate(top_k_indices[1:]):  # 跳過第一個（自己）
            rec_artist = node_encoder.inverse_transform([idx])[0]
            recommendations.append(f"{rec_artist} ({top_k_values[i]:.2f})")
        
        # 獲取該藝人的實際合作者
        actual_collaborators = test_df[test_df['source'] == artist]['target'].unique()
        
        source_group = artist.split('_')[0]
        # 為每個實際合作者創建一條預測記錄
        for collaborator in actual_collaborators:
            collaborator_group = collaborator.split('_')[0]
            if source_group == collaborator_group:
                continue
            
            predictions.append({
                'source': artist,
                'collaborator': collaborator,
                'recommend_idol': ', '.join(recommendations)
            })
    
    # 創建預測結果 DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # 保存為 CSV 文件
    output_dir = 'predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, 'predictions_v.csv')
    predictions_df.to_csv(output_file, index=False)
    print(f"\n預測結果已保存至: {output_file}")
    
    return predictions_df

def main():
    # 載入數據
    df = pd.read_csv('data/collaboration_videos.csv')
    
    # 將時間戳轉換為浮點數
    df['timestamp'] = df['timestamp'].astype(float)
    
    # 設置分割時間戳（2025-01-01 的 Unix 時間戳）
    split_timestamp = datetime(2025, 1, 1).timestamp()
    
    # test_df = df[df['timestamp'] >= split_timestamp]
    
    node_encoder = get_node_encoder()
    
    # 準備圖數據    
    train_data = torch.load('gnn_best/model＿without.pth')
    
    # 初始化模型
    model = GNNRecommender(num_features=train_data['num_features']).to('cuda')
    
    # 載入訓練好的模型權重
    model.load_state_dict(train_data['model'])
    model.eval()  # 設置為評估模式
    
    # 評估模型
    print("\n開始評估模型...")
    metrics = evaluate_model(model, df, node_encoder)
    
    print("\n評估結果：")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Hit@10: {metrics['hit@10']:.4f}")
    
    # 生成預測結果
    # print("\n生成預測結果...")
    # predictions_df = generate_predictions(model, test_data, test_df, node_encoder)
    
    return

if __name__ == "__main__":
    main() 