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
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

def get_node_encoder():
    node_encoder = LabelEncoder()
    df = pd.read_csv("/home/bl515-ml/Documents/shaio_jie/sma/Kpop_challenge_analyze/data/node_list.csv")
    node_encoder.fit(df['node'].unique())
    
    return node_encoder

def get_collaboration_df(remove_same_group=False):
    collaboration_df = pd.read_csv("/home/bl515-ml/Documents/shaio_jie/sma/Kpop_challenge_analyze/data/collaboration_videos.csv")
    if remove_same_group:
        collaboration_df = collaboration_df[collaboration_df['source'].str.split('_').str[0] != collaboration_df['target'].str.split('_').str[0]]
    return collaboration_df

def prepare_temporal_graph_data(collaboration_df, node_encoder, window_size=60, remove_same_group=False, training=True):
    """
    準備時序圖神經網絡所需的數據，使用滑動窗口
    
    參數:
    - collaboration_df: 合作關係數據框
    - node_encoder: 節點編碼器
    - window_size: 時間窗口大小（天）
    
    返回:
    - DynamicGraphTemporalSignal 對象
    """
    # 獲取所有節點
    num_nodes = node_encoder.classes_.shape[0]
    x = torch.eye(num_nodes)  # 使用 one-hot 編碼作為初始特徵
    
    # 按時間排序數據
    collaboration_df = collaboration_df.sort_values('timestamp')
    
    if remove_same_group:
        collaboration_df = collaboration_df[collaboration_df['source'].str.split('_').str[0] != collaboration_df['target'].str.split('_').str[0]]
    
    # 準備存儲每個時間窗口的數據
    edge_indices = []
    edge_weights = []
    edge_features = []
    node_features = []
    
    
    if training:
      itter = collaboration_df[collaboration_df['timestamp'] < datetime(2025, 1, 1).timestamp()]
    else:
      itter = collaboration_df[collaboration_df['timestamp'] >= datetime(2025, 1, 1).timestamp()]
    
    # 使用滑動窗口處理數據
    for _, row in itter.iterrows():
        start_time = row['timestamp'] - window_size * 24 * 3600
        # 獲取當前窗口內的數據
        window_data = collaboration_df[
            (collaboration_df['timestamp'] >= start_time) & 
            (collaboration_df['timestamp'] < row['timestamp'])
        ]
        
        if len(window_data) > 0:
            # 創建邊索引和特徵
            edge_index = []
            edge_feature = []
            
            for _, row in window_data.iterrows():
                artist1_idx = node_encoder.transform([row['source']])[0]
                artist2_idx = node_encoder.transform([row['target']])[0]
                
                # 添加雙向邊
                edge_index.append([artist1_idx, artist2_idx])
                edge_index.append([artist2_idx, artist1_idx])
                
                # 計算邊特徵
                views = row['views'] if 'views' in row else 0
                likes = row['likes'] if 'likes' in row else 0
                comments = row['comments'] if 'comments' in row else 0

                # 組合特徵
                feature = [ views, likes, comments]
                edge_feature.append(feature)
                edge_feature.append(feature)  # 雙向邊使用相同的特徵
            
            # 轉換為張量
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_feature = torch.tensor(edge_feature, dtype=torch.float)
            
            # 添加到列表中
            edge_indices.append(edge_index)
            edge_features.append(edge_feature)
            node_features.append(x)  # 使用相同的節點特徵
            
    
    # 創建 DynamicGraphTemporalSignal 對象
    temporal_signal = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_features,
        features=node_features,
        targets=node_features
    )
    
    return temporal_signal

def prepare_graph_data(node_encoder):
    """
    準備圖神經網絡所需的數據，並根據時間戳分割訓練集和測試集
    """
    num_nodes = node_encoder.classes_.shape[0]
    x = torch.eye(num_nodes)  # 使用 one-hot 編碼作為初始特徵
    
    # 創建訓練集的邊索引和屬性
    edge_index = []
    edge_attr = []
    edge_features = []
    
    collaboration_df = get_collaboration_df(remove_same_group=True)
    
    for _, row in collaboration_df.iterrows():
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
    