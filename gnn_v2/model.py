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
    def __init__(self, num_features, hidden_channels=128, embedding_dim=64):
        super(GNNRecommender, self).__init__()
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            GATConv(num_features, hidden_channels, heads=4),
            torch.nn.LayerNorm(hidden_channels * 4),
            torch.nn.ELU(),
            torch.nn.Dropout(0.2),
            
            GATConv(hidden_channels * 4, hidden_channels, heads=2),
            torch.nn.LayerNorm(hidden_channels * 2),
            torch.nn.ELU(),
            torch.nn.Dropout(0.2),
            
            GATConv(hidden_channels * 2, embedding_dim),
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.ELU()
        )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ELU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(hidden_channels, num_features),
            torch.nn.LayerNorm(num_features),
            torch.nn.ELU()
        )
        
    def encode(self, x, edge_index, edge_attr=None):
        # 提取節點嵌入
        for layer in self.encoder:
            if isinstance(layer, GATConv):
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x)
        return x
    
    def decode(self, x):
        # 解碼節點嵌入
        for layer in self.decoder:
            x = layer(x)
        return x
    
    def forward(self, x, edge_index, edge_attr=None):
        # 編碼
        embeddings = self.encode(x, edge_index, edge_attr)
        # 解碼
        reconstructed = self.decode(embeddings)
        return reconstructed
    
    def get_embedding(self, x, edge_index, edge_attr=None):
        # 獲取節點嵌入
        with torch.no_grad():
            return self.encode(x, edge_index, edge_attr)
    