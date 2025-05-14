import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score

class GNNRecommender(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=64):
        super(GNNRecommender, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_features)  # 最後一層輸出維度改回原始特徵維度
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # 輸出維度與輸入相同
        return x

def prepare_graph_data(collaboration_df, time_window=60, split_timestamp=None):
    """
    準備圖神經網絡所需的數據，並根據時間戳分割訓練集和測試集
    """
    # 創建節點編碼器
    node_encoder = LabelEncoder()
    
    # 獲取所有唯一的藝人
    all_artists = pd.concat([collaboration_df['source'], collaboration_df['target']]).unique()
    node_encoder.fit(all_artists)
    
    # 創建節點特徵矩陣
    num_nodes = len(all_artists)
    x = torch.eye(num_nodes)  # 使用 one-hot 編碼作為初始特徵
    
    # 分割訓練集和測試集
    if split_timestamp:
        train_df = collaboration_df[collaboration_df['timestamp'] < split_timestamp]
        test_df = collaboration_df[collaboration_df['timestamp'] >= split_timestamp]
    else:
        train_df = collaboration_df
        test_df = pd.DataFrame()
    
    # 創建訓練集的邊索引和屬性
    train_edge_index = []
    train_edge_attr = []
    
    # 處理訓練數據
    for _, row in train_df.iterrows():
        artist1_idx = node_encoder.transform([row['source']])[0]
        artist2_idx = node_encoder.transform([row['target']])[0]
        
        # 添加雙向邊
        train_edge_index.append([artist1_idx, artist2_idx])
        train_edge_index.append([artist2_idx, artist1_idx])
        
        # 邊的權重（這裡使用時間衰減）
        current_time = datetime.now().timestamp()
        time_diff = (current_time - row['timestamp']) / (24 * 3600)  # 轉換為天
        weight = np.exp(-time_diff / time_window)
        train_edge_attr.extend([weight, weight])
    
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()
    train_edge_attr = torch.tensor(train_edge_attr, dtype=torch.float)
    
    train_data = Data(x=x, edge_index=train_edge_index, edge_attr=train_edge_attr)
    
    # 如果有測試集，也創建測試數據
    test_data = None
    if not test_df.empty:
        test_edge_index = []
        test_edge_attr = []
        
        for _, row in test_df.iterrows():
            artist1_idx = node_encoder.transform([row['source']])[0]
            artist2_idx = node_encoder.transform([row['target']])[0]
            
            test_edge_index.append([artist1_idx, artist2_idx])
            test_edge_index.append([artist2_idx, artist1_idx])
            
            current_time = datetime.now().timestamp()
            time_diff = (current_time - row['timestamp']) / (24 * 3600)  # 轉換為天
            weight = np.exp(-time_diff / time_window)
            test_edge_attr.extend([weight, weight])
        
        test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).t().contiguous()
        test_edge_attr = torch.tensor(test_edge_attr, dtype=torch.float)
        
        test_data = Data(x=x, edge_index=test_edge_index, edge_attr=test_edge_attr)
    
    return train_data, test_data, node_encoder, test_df

def train_model(model, data, optimizer, epochs=200):
    """
    訓練 GNN 模型
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # 使用重建損失，確保輸入和輸出維度相同
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

def evaluate_model(model, test_data, test_df, node_encoder, top_k=5):
    """
    評估模型在測試集上的表現
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(test_data.x, test_data.edge_index)
    
    # 計算每個藝人的推薦
    all_recommendations = {}
    for artist in test_df['source'].unique():
        artist_idx = node_encoder.transform([artist])[0]
        artist_embedding = embeddings[artist_idx]
        
        similarities = F.cosine_similarity(artist_embedding.unsqueeze(0), embeddings)
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k+1)
        
        recommendations = []
        for idx in top_k_indices[1:]:
            rec_artist = node_encoder.inverse_transform([idx])[0]
            recommendations.append(rec_artist)
        
        all_recommendations[artist] = recommendations
    
    # 計算評估指標
    true_positives = 0
    total_recommendations = 0
    total_actual_collaborations = 0
    reciprocal_ranks = []  # 用於計算 MRR
    
    for _, row in test_df.iterrows():
        source = row['source']
        target = row['target']
        
        if source in all_recommendations:
            recommendations = all_recommendations[source]
            if target in recommendations:
                true_positives += 1
                # 計算 MRR
                rank = recommendations.index(target) + 1
                reciprocal_ranks.append(1.0 / rank)
            total_recommendations += len(recommendations)
            total_actual_collaborations += 1
    
    precision = true_positives / total_recommendations if total_recommendations > 0 else 0
    recall = true_positives / total_actual_collaborations if total_actual_collaborations > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mrr': mrr
    }

def get_recommendations(model, data, node_encoder, artist_name, top_k=5):
    """
    為指定藝人獲取推薦
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        
    # 獲取目標藝人的嵌入
    artist_idx = node_encoder.transform([artist_name])[0]
    artist_embedding = embeddings[artist_idx]
    
    # 計算與其他藝人的相似度
    similarities = F.cosine_similarity(artist_embedding.unsqueeze(0), embeddings)
    
    # 獲取 top-k 推薦
    top_k_values, top_k_indices = torch.topk(similarities, k=top_k+1)
    
    # 轉換回藝人名字
    recommendations = []
    for idx in top_k_indices[1:]:  # 跳過第一個（自己）
        artist_name = node_encoder.inverse_transform([idx])[0]
        recommendations.append(artist_name)
    
    return recommendations

def main():
    # 讀取數據
    df = pd.read_csv('data/collaboration_videos.csv')
    
    # 將時間戳轉換為浮點數
    df['timestamp'] = df['timestamp'].astype(float)
    
    # 設置分割時間戳（2025-01-01 的 Unix 時間戳）
    split_timestamp = datetime(2025, 1, 1).timestamp()
    
    # 準備圖數據
    train_data, test_data, node_encoder, test_df = prepare_graph_data(df, split_timestamp=split_timestamp)
    
    # 初始化模型
    model = GNNRecommender(num_features=train_data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 訓練模型
    print("開始訓練模型...")
    train_model(model, train_data, optimizer)
    
    # 評估模型
    print("\n開始評估模型...")
    metrics = evaluate_model(model, test_data, test_df, node_encoder)
    print("\n評估結果：")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    
    # 為一些藝人生成推薦
    print("\n為測試集中的一些藝人生成推薦：")
    test_artists = test_df['source'].unique()[:3]  # 取前三個藝人作為示例
    for artist in test_artists:
        recommendations = get_recommendations(model, test_data, node_encoder, artist)
        print(f"\n為 {artist} 的推薦：")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    main() 