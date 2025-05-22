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
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from util import get_node_encoder, prepare_temporal_graph_data

# 加載數據
df = pd.read_csv("/home/bl515-ml/Documents/shaio_jie/sma/Kpop_challenge_analyze/data/collaboration_videos.csv")
df['timestamp'] = df['timestamp'].astype(float)

# 獲取節點編碼器
node_encoder = get_node_encoder()

# 創建時序數據加載器
temporal_signal = prepare_temporal_graph_data(df, node_encoder, window_size=60, remove_same_group=True)

# 打印數據加載器信息
print(f"時間窗口數量: {len(temporal_signal)}")
print(f"節點數量: {temporal_signal[0].x.shape[0]}")
print(f"第一個時間窗口的邊數量: {temporal_signal[0].edge_index.shape[1]}")
print(f"邊特徵維度: {temporal_signal[0].edge_attr.shape[1]}")

# 遍歷時間窗口
for i, snapshot in enumerate(temporal_signal):
    print(f"\n時間窗口 {i}:")
    print(f"邊數量: {snapshot.edge_index.shape[1]}")
    print(f"邊特徵形狀: {snapshot.edge_attr.shape}") 