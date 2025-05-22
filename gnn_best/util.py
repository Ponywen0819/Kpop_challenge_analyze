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

def get_node_encoder():
    node_encoder = LabelEncoder()
    df = pd.read_csv("/home/bl515-ml/Documents/shaio_jie/sma/Kpop_challenge_analyze/data/node_list.csv")
    node_encoder.fit(df['node'].unique())
    
    return node_encoder

def prepare_graph_data(collaboration_df, node_encoder, start_timestamp=None, end_timestamp=None):
    """
    準備圖神經網絡所需的數據，並根據時間戳分割訓練集和測試集
    """
    num_nodes = node_encoder.classes_.shape[0]
    x = torch.eye(num_nodes)  # 使用 one-hot 編碼作為初始特徵
    
    # 創建訓練集的邊索引和屬性
    edge_index = []
    edge_attr = []
    edge_features = []
    
    for _, row in collaboration_df.iterrows():
      source_group = row['source'].split('_')[0]
      target_group = row['target'].split('_')[0]
      
      if source_group == target_group:
        continue
      
      if start_timestamp and row['timestamp'] < start_timestamp:
        continue
      if end_timestamp and row['timestamp'] > end_timestamp:
        continue
      
      artist1_idx = node_encoder.transform([row['source']])[0]
      artist2_idx = node_encoder.transform([row['target']])[0]
      
      # 添加雙向邊
      edge_index.append([artist1_idx, artist2_idx])
      # edge_index.append([artist2_idx, artist1_idx])
        
      # 使用 views, likes, comments 作為額外特徵
      views = row['views'] if 'views' in row else 0
      likes = row['likes'] if 'likes' in row else 0
      comments = row['comments'] if 'comments' in row else 0
      
      # 組合所有特徵
      edge_feature = [views, likes, comments]
      edge_features.append(edge_feature) # 雙向邊使用相同的特徵
      # edge_features.append(edge_feature)
      
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    