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
from gnn_best.model import GNNRecommender
from gnn_best.util import get_node_encoder, prepare_graph_data

def train_model(model, data, optimizer, epochs=300):
    """
    訓練 GNN 模型
    """
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # 使用多任務學習損失
        reconstruction_loss = F.cross_entropy(out, data.x)
        
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
  
def main():
    # 讀取數據
    df = pd.read_csv('data/collaboration_videos.csv')
    
    # 將時間戳轉換為浮點數
    df['timestamp'] = df['timestamp'].astype(float)
    
    # 設置分割時間戳（2025-01-01 的 Unix 時間戳）
    split_timestamp = datetime(2025, 1, 1).timestamp()
    
    node_encoder = get_node_encoder()
    
    train_df = df[df['timestamp'] < split_timestamp]
    
    # 準備圖數據
    train_data= prepare_graph_data(train_df, node_encoder)
    
    # 初始化模型
    model = GNNRecommender(num_features=train_data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 訓練模型
    print("開始訓練模型...")
    train_model(model, train_data, optimizer)
    
    # 儲存模型
    torch.save({
        'model': model.state_dict(),
        'num_features': train_data.x.size(1)
    }, 'gnn_best/model＿without.pth')
    
if __name__ == "__main__":
    main() 