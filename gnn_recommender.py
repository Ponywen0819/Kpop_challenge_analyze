import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score
import os

class GNNRecommender(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=128):
        super(GNNRecommender, self).__init__()
        # 使用 GAT 層來增加注意力機制
        self.conv1 = GATConv(num_features, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=2)
        self.conv3 = GATConv(hidden_channels * 2, num_features)
        
        # 添加層標準化
        self.layer_norm1 = torch.nn.LayerNorm(hidden_channels * 4)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_channels * 2)
        
    def forward(self, x, edge_index, edge_attr=None):
        # 第一層
        x = self.conv1(x, edge_index, edge_attr)
        x = self.layer_norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 第二層
        x = self.conv2(x, edge_index, edge_attr)
        x = self.layer_norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 第三層
        x = self.conv3(x, edge_index, edge_attr)
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
    
    # 準備邊特徵
    edge_scaler = MinMaxScaler()
    edge_features = []
    
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
        
        # 計算邊特徵
        current_time = datetime.now().timestamp()
        time_diff = (current_time - row['timestamp']) / (24 * 3600)  # 轉換為天
        time_weight = np.exp(-time_diff / time_window)
        
        # 使用 views, likes, comments 作為額外特徵
        views = row['views'] if 'views' in row else 0
        likes = row['likes'] if 'likes' in row else 0
        comments = row['comments'] if 'comments' in row else 0
        
        # 組合所有特徵
        edge_feature = [time_weight, views, likes, comments]
        edge_features.append(edge_feature)
        edge_features.append(edge_feature)  # 雙向邊使用相同的特徵
    
    # 標準化邊特徵
    edge_features = edge_scaler.fit_transform(edge_features)
    train_edge_attr = torch.tensor(edge_features, dtype=torch.float)
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()
    
    train_data = Data(x=x, edge_index=train_edge_index, edge_attr=train_edge_attr)
    
    # 如果有測試集，也創建測試數據
    test_data = None
    if not test_df.empty:
        test_edge_index = []
        test_edge_attr = []
        test_edge_features = []
        
        for _, row in test_df.iterrows():
            artist1_idx = node_encoder.transform([row['source']])[0]
            artist2_idx = node_encoder.transform([row['target']])[0]
            
            test_edge_index.append([artist1_idx, artist2_idx])
            test_edge_index.append([artist2_idx, artist1_idx])
            
            current_time = datetime.now().timestamp()
            time_diff = (current_time - row['timestamp']) / (24 * 3600)
            time_weight = np.exp(-time_diff / time_window)
            
            views = row['views'] if 'views' in row else 0
            likes = row['likes'] if 'likes' in row else 0
            comments = row['comments'] if 'comments' in row else 0
            
            edge_feature = [time_weight, views, likes, comments]
            test_edge_features.append(edge_feature)
            test_edge_features.append(edge_feature)
        
        test_edge_features = edge_scaler.transform(test_edge_features)
        test_edge_attr = torch.tensor(test_edge_features, dtype=torch.float)
        test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).t().contiguous()
        
        test_data = Data(x=x, edge_index=test_edge_index, edge_attr=test_edge_attr)
    
    return train_data, test_data, node_encoder, test_df

def train_model(model, data, optimizer, epochs=200):
    """
    訓練 GNN 模型
    """
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # 使用多任務學習損失
        reconstruction_loss = F.mse_loss(out, data.x)
        
        # # 添加對比學習損失
        # pos_mask = (data.edge_index[0] != data.edge_index[1])
        # pos_edges = data.edge_index[:, pos_mask]
        # neg_edges = generate_negative_edges(data.edge_index, data.x.size(0))
        
        # pos_scores = torch.sum(out[pos_edges[0]] * out[pos_edges[1]], dim=1)
        # neg_scores = torch.sum(out[neg_edges[0]] * out[neg_edges[1]], dim=1)
        
        # contrastive_loss = -torch.mean(
        #     torch.log(torch.sigmoid(pos_scores)) + 
        #     torch.log(1 - torch.sigmoid(neg_scores))
        # )
        
        # # 組合損失
        # loss = reconstruction_loss + 0.5 * contrastive_loss
        loss = reconstruction_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

def generate_negative_edges(edge_index, num_nodes, num_neg_samples=None):
    """
    生成負樣本邊
    """
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        src = torch.randint(0, num_nodes, (1,))
        dst = torch.randint(0, num_nodes, (1,))
        if src != dst and not is_edge_exists(src, dst, edge_index):
            neg_edges.append([src.item(), dst.item()])
    
    return torch.tensor(neg_edges, dtype=torch.long).t()

def is_edge_exists(src, dst, edge_index):
    """
    檢查邊是否存在
    """
    return torch.any((edge_index[0] == src) & (edge_index[1] == dst))

def evaluate_model(model, test_data, test_df, node_encoder, top_k=10):
    """
    評估模型在測試集上的表現
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(test_data.x, test_data.edge_index, test_data.edge_attr)
    
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
    reciprocal_ranks = []
    hits_at_k = 0  # 用於計算 Hit@K
    
    for _, row in test_df.iterrows():
        source = row['source']
        target = row['target']
        
        if source in all_recommendations:
            recommendations = all_recommendations[source]
            if target in recommendations:
                true_positives += 1
                rank = recommendations.index(target) + 1
                reciprocal_ranks.append(1.0 / rank)
                if rank <= top_k:
                    hits_at_k += 1
            total_recommendations += len(recommendations)
            total_actual_collaborations += 1
    
    precision = true_positives / total_recommendations if total_recommendations > 0 else 0
    recall = true_positives / total_actual_collaborations if total_actual_collaborations > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    hit_at_k = hits_at_k / total_actual_collaborations if total_actual_collaborations > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mrr': mrr,
        f'hit@{top_k}': hit_at_k
    }

def get_recommendations(model, data, node_encoder, artist_name, top_k=5):
    """
    為指定藝人獲取推薦
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_attr)
        
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

def generate_predictions(model, test_data, test_df, node_encoder, top_k=10):
    """
    生成預測結果並保存為 CSV 文件
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(test_data.x, test_data.edge_index, test_data.edge_attr)
    
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
            recommendations.append(F"{rec_artist} ({top_k_values[i]:.4f})")
        
        # 獲取該藝人的實際合作者
        actual_collaborators = test_df[test_df['source'] == artist]['target'].unique()
        
        # 為每個實際合作者創建一條預測記錄
        for collaborator in actual_collaborators:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
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
    print(f"Hit@10: {metrics['hit@10']:.4f}")
    
    # 生成預測結果
    print("\n生成預測結果...")
    predictions_df = generate_predictions(model, test_data, test_df, node_encoder)
    
    # 顯示一些示例預測
    print("\n預測示例：")
    for _, row in predictions_df.head(3).iterrows():
        print(f"\n藝人: {row['source']}")
        print(f"實際合作者: {row['collaborator']}")
        print(f"推薦列表: {row['recommend_idol']}")

if __name__ == "__main__":
    main() 