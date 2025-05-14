import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

class GNNRecommender(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads=4, edge_features_dim=None):
        super(GNNRecommender, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1)
        self.final_layer = torch.nn.Linear(hidden_channels, num_features)  # 添加最終層
        
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
        x = self.final_layer(x)  # 將輸出映射回原始特徵空間
        return x

def load_data():
    # 加載節點數據
    nodes_df = pd.read_csv('data/node_list.csv')
    
    # 創建節點ID到索引的映射
    node_to_idx = {node: idx for idx, node in enumerate(nodes_df['node'])}
    
    # 加載合作關係數據
    collab_df = pd.read_csv('data/collaboration_videos.csv')
    
    # 創建節點特徵矩陣
    num_nodes = len(nodes_df)
    x = torch.eye(num_nodes)  # 使用one-hot編碼作為初始特徵
    
    # 創建邊索引和邊特徵
    edge_index = []
    edge_features = []
    
    # 標準化數值特徵
    scaler = StandardScaler()
    numeric_features = ['views', 'likes', 'comments']
    collab_df[numeric_features] = scaler.fit_transform(collab_df[numeric_features])
    
    # 處理時間特徵
    if 'timestamp' in collab_df.columns:
        collab_df['timestamp'] = pd.to_datetime(collab_df['timestamp'])
        collab_df['time_feature'] = (collab_df['timestamp'] - collab_df['timestamp'].min()).dt.total_seconds()
        collab_df['time_feature'] = scaler.fit_transform(collab_df[['time_feature']])
    
    for _, row in collab_df.iterrows():
        source = node_to_idx[row['source']]
        target = node_to_idx[row['target']]
        edge_index.append([source, target])
        
        # 收集邊特徵
        edge_feature = []
        for feature in numeric_features:
            if feature in collab_df.columns:
                edge_feature.append(float(row[feature]))
        if 'time_feature' in collab_df.columns:
            edge_feature.append(float(row['time_feature']))
            
        edge_features.append(edge_feature)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_to_idx

def calculate_metrics(model, data, test_edges, k_values=[5,10]):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_attr)
        
        # 計算所有節點對之間的相似度
        similarities = torch.matmul(embeddings, embeddings.t())
        
        # 初始化評估指標
        hit_rates = {k: 0.0 for k in k_values}
        mrr = 0.0
        total_samples = len(test_edges)
        
        for source, target in test_edges:
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

def train_model(model, data, optimizer, epochs=200):
    # 將邊分成訓練集和測試集
    num_edges = data.edge_index.size(1)
    indices = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_edge_index = data.edge_index[:, train_indices]
    test_edges = data.edge_index[:, test_indices].t().tolist()
    
    # 創建訓練數據
    train_data = Data(x=data.x, edge_index=train_edge_index, edge_attr=data.edge_attr[train_indices])
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = F.mse_loss(out, train_data.x)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # 計算評估指標
            hit_rates, mrr = calculate_metrics(model, data, test_edges)
            print(f'Epoch {epoch}:')
            print(f'Loss = {loss.item():.4f}')
            print(f'MRR = {mrr:.4f}')
            for k, hr in hit_rates.items():
                print(f'Hit Rate@{k} = {hr:.4f}')
            print('-------------------')

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
    data, node_to_idx = load_data()
    
    # 初始化模型
    model = GNNRecommender(
        num_features=data.x.size(1),
        hidden_channels=64,
        edge_features_dim=data.edge_attr.size(1) if hasattr(data, 'edge_attr') else None
    )
    
    # 設置優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 訓練模型
    train_model(model, data, optimizer)
    
    # 示例：為某個節點獲取推薦
    test_node = 0  # 示例節點索引
    recommendations = get_recommendations(model, data, test_node)
    print(f"Recommendations for node {test_node}: {recommendations}")

if __name__ == "__main__":
    main() 