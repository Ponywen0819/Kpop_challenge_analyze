import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt

class GNNRecommender(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, edge_features_dim=None):
        super(GNNRecommender, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1)
        self.final_layer = torch.nn.Linear(hidden_channels, num_features)
        
        if edge_features_dim is not None:
            self.edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(edge_features_dim, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is not None and hasattr(self, 'edge_mlp'):
            edge_attr = self.edge_mlp(edge_attr)
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.final_layer(x)
        return x

def load_data():
    # 加載節點數據
    nodes_df = pd.read_csv('data/node_list.csv')
    
    # 創建節點ID到索引的映射
    node_to_idx = {node: idx for idx, node in enumerate(nodes_df['node'])}
    
    # 加載合作關係數據
    collab_df = pd.read_csv('data/collaboration_videos.csv')
    
    # 將時間戳轉換為datetime
    collab_df['timestamp'] = pd.to_datetime(collab_df['timestamp'], unit='s')
    
    # 設置測試集時間閾值
    test_threshold = pd.Timestamp('2025-01-01')
    
    # 分割訓練集和測試集
    train_df = collab_df[collab_df['timestamp'] < test_threshold]
    test_df = collab_df[collab_df['timestamp'] >= test_threshold]
    
    print(f"訓練集大小: {len(train_df)}")
    print(f"測試集大小: {len(test_df)}")
    print(f"訓練集時間範圍: {train_df['timestamp'].min()} 到 {train_df['timestamp'].max()}")
    print(f"測試集時間範圍: {test_df['timestamp'].min()} 到 {test_df['timestamp'].max()}")
    
    # 創建節點特徵矩陣
    num_nodes = len(nodes_df)
    x = torch.eye(num_nodes)
    
    # 標準化數值特徵
    scaler = StandardScaler()
    numeric_features = ['views', 'likes', 'comments']
    train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])
    
    # 處理時間特徵
    train_df['time_feature'] = (train_df['timestamp'] - train_df['timestamp'].min()).dt.total_seconds()
    test_df['time_feature'] = (test_df['timestamp'] - train_df['timestamp'].min()).dt.total_seconds()
    
    train_df['time_feature'] = scaler.fit_transform(train_df[['time_feature']])
    test_df['time_feature'] = scaler.transform(test_df[['time_feature']])
    
    # 創建訓練集邊索引和邊特徵
    train_edge_index = []
    train_edge_features = []
    
    for _, row in train_df.iterrows():
        source = node_to_idx[row['source']]
        target = node_to_idx[row['target']]
        train_edge_index.append([source, target])
        
        edge_feature = []
        for feature in numeric_features:
            edge_feature.append(float(row[feature]))
        edge_feature.append(float(row['time_feature']))
        train_edge_features.append(edge_feature)
    
    # 創建測試集邊索引和邊特徵
    test_edges = []
    test_edge_features = []
    
    for _, row in test_df.iterrows():
        source = node_to_idx[row['source']]
        target = node_to_idx[row['target']]
        test_edges.append([source, target])
        
        edge_feature = []
        for feature in numeric_features:
            edge_feature.append(float(row[feature]))
        edge_feature.append(float(row['time_feature']))
        test_edge_features.append(edge_feature)
    
    # 轉換為張量
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()
    train_edge_attr = torch.tensor(train_edge_features, dtype=torch.float)
    test_edges = torch.tensor(test_edges, dtype=torch.long)
    test_edge_attr = torch.tensor(test_edge_features, dtype=torch.float)
    
    # 創建訓練數據
    train_data = Data(x=x, edge_index=train_edge_index, edge_attr=train_edge_attr)
    
    return train_data, test_edges, test_edge_attr, node_to_idx

def calculate_metrics(model, data, test_edges, test_edge_attr, k_values=[1, 3, 5, 10]):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_attr)
        
        # 計算所有節點對之間的相似度
        similarities = torch.matmul(embeddings, embeddings.t())
        
        # 初始化評估指標
        hit_rates = {k: 0.0 for k in k_values}
        mrr = 0.0
        total_samples = len(test_edges)
        
        for i, (source, target) in enumerate(test_edges):
            # 獲取源節點的所有預測分數
            pred_scores = similarities[source]
            
            # 排除自身
            pred_scores[source] = float('-inf')
            
            # 獲取排序後的索引
            _, indices = torch.sort(pred_scores, descending=True)
            
            # 計算目標節點的排名
            rank = (indices == target).nonzero().item() + 1
            
            # 更新 MRR
            mrr += 1.0 / rank
            
            # 更新 Hit Rate
            for k in k_values:
                if rank <= k:
                    hit_rates[k] += 1.0
        
        # 計算平均值
        mrr /= total_samples
        for k in k_values:
            hit_rates[k] /= total_samples
            
        return hit_rates, mrr

def evaluate_test_set(model, train_data, test_edges, test_edge_attr, node_to_idx, k_values=[1, 3, 5, 10]):
    """
    對測試集進行詳細的效能評估
    """
    model.eval()
    with torch.no_grad():
        # 獲取節點嵌入
        embeddings = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        
        # 計算所有節點對之間的相似度
        similarities = torch.matmul(embeddings, embeddings.t())
        
        # 初始化評估指標
        hit_rates = {k: 0.0 for k in k_values}
        mrr = 0.0
        total_samples = len(test_edges)
        
        # 用於繪製ROC曲線的數據
        all_scores = []
        all_labels = []
        
        # 用於記錄每個K值的詳細結果
        detailed_results = {k: {'hits': [], 'misses': []} for k in k_values}
        
        for i, (source, target) in enumerate(test_edges):
            # 獲取源節點的所有預測分數
            pred_scores = similarities[source]
            
            # 排除自身
            pred_scores[source] = float('-inf')
            
            # 獲取排序後的索引和分數
            scores, indices = torch.sort(pred_scores, descending=True)
            
            # 計算目標節點的排名
            rank = (indices == target).nonzero().item() + 1
            
            # 更新 MRR
            mrr += 1.0 / rank
            
            # 更新 Hit Rate 並記錄詳細結果
            for k in k_values:
                if rank <= k:
                    hit_rates[k] += 1.0
                    detailed_results[k]['hits'].append((source.item(), target.item(), rank))
                else:
                    detailed_results[k]['misses'].append((source.item(), target.item(), rank))
            
            # 收集ROC曲線數據
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend([1 if idx == target else 0 for idx in indices])
        
        # 計算平均值
        mrr /= total_samples
        for k in k_values:
            hit_rates[k] /= total_samples
        
        # 打印詳細結果
        print("\n=== 測試集評估結果 ===")
        print(f"總測試樣本數: {total_samples}")
        print(f"MRR: {mrr:.4f}")
        for k in k_values:
            print(f"\nTop-{k} 結果:")
            print(f"Hit Rate: {hit_rates[k]:.4f}")
            print(f"命中數: {len(detailed_results[k]['hits'])}")
            print(f"未命中數: {len(detailed_results[k]['misses'])}")
            
        return hit_rates, mrr, detailed_results

def train_model(model, train_data, test_edges, test_edge_attr, optimizer, epochs=200):
    model.train()
    best_mrr = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = F.mse_loss(out, train_data.x)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # 計算評估指標
            hit_rates, mrr = calculate_metrics(model, train_data, test_edges, test_edge_attr)
            print(f'Epoch {epoch}:')
            print(f'Loss = {loss.item():.4f}')
            print(f'MRR = {mrr:.4f}')
            for k, hr in hit_rates.items():
                print(f'Hit Rate@{k} = {hr:.4f}')
            print('-------------------')
            
            # 保存最佳模型
            if mrr > best_mrr:
                best_mrr = mrr
                best_model_state = model.state_dict().copy()
    
    # 載入最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def get_recommendations(model, data, node_idx, top_k=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_attr)
        node_embedding = embeddings[node_idx]
        
        # 計算與其他節點的相似度
        similarities = F.cosine_similarity(node_embedding.unsqueeze(0), embeddings)
        
        # 獲取top-k推薦
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k+1)
        
        return top_k_indices[1:].tolist()  # 排除自身

def main():
    # 加載數據
    train_data, test_edges, test_edge_attr, node_to_idx = load_data()
    
    # 初始化模型
    model = GNNRecommender(
        num_features=train_data.x.size(1),
        hidden_channels=64,
        edge_features_dim=train_data.edge_attr.size(1) if hasattr(train_data, 'edge_attr') else None
    )
    
    # 設置優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 訓練模型
    model = train_model(model, train_data, test_edges, test_edge_attr, optimizer)
    
    # 在測試集上進行詳細評估
    hit_rates, mrr, detailed_results = evaluate_test_set(
        model, train_data, test_edges, test_edge_attr, node_to_idx
    )
    
    # 示例：為某個節點獲取推薦
    test_node = 0  # 示例節點索引
    recommendations = get_recommendations(model, train_data, test_node)
    print(f"\nRecommendations for node {test_node}: {recommendations}")

if __name__ == "__main__":
    main() 