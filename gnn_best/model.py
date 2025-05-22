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
        self.layer_norm3 = torch.nn.LayerNorm(num_features)
        
        self.forward_layer = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features),
            torch.nn.LayerNorm(num_features),
            torch.nn.ELU()
        )
        
        
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
        x = self.layer_norm3(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.forward_layer(x)
        
        return x
    
    def get_embedding(self, x, edge_index, edge_attr=None):
        # 第一層
        x = self.conv1(x, edge_index, edge_attr)
        x = self.layer_norm1(x)
        x = F.elu(x)
        
        # 第二層
        x = self.conv2(x, edge_index, edge_attr)
        x = self.layer_norm2(x)
        x = F.elu(x)
        
        # 第三層
        x = self.conv3(x, edge_index, edge_attr)
        x = self.layer_norm3(x)
        x = F.elu(x)
        
        return x
    