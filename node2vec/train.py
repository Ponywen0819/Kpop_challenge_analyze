import numpy as np
import networkx as nx
import node2vec
import json
import os
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from gensim.models import Word2Vec
from torch.utils.data import Dataset
import random
import node2vecClassifier

# 获取当前文件的目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
ROOT_DIR = os.path.dirname(CURRENT_DIR)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_collaboration_data():
    """加载已清洗的合作视频数据"""
    data_path = os.path.join(ROOT_DIR, "data", "collaboration_videos.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def build_collaboration_graph():
    """建立合作网络图"""
    G = nx.DiGraph()
    
    collaboration_data = load_collaboration_data()
    
    # 建立節點
    for video in collaboration_data:
        source_member = video['source']
        target_member = video['target']
        G.add_node(f"{source_member[0]}_{source_member[1]}")
        G.add_node(f"{target_member[0]}_{target_member[1]}")
    
    sliced_collaboration_data = [d for d in collaboration_data if d['timestamp'] <  datetime(2025, 1, 1).timestamp()]
    
    # 建立邊
    for video in sliced_collaboration_data:
        source_member = video['source']
        target_member = video['target']

        G.add_edge(f"{source_member[0]}_{source_member[1]}", f"{target_member[0]}_{target_member[1]}")
    
    # 计算边的权重
    idol_values = defaultdict(list)
    for video in sliced_collaboration_data:
        source_member = video['source']
        target_member = video['target']
        effectiveness = video['views'] * 0.5 + video['likes'] * 0.3 + video['comments'] * 0.2
        
        combination = (f"{source_member[0]}_{source_member[1]}", f"{target_member[0]}_{target_member[1]}")
        idol_values[combination].append(effectiveness)
        
    # 计算平均效益
    idol_avg = {name: sum(vals) / len(vals) for name, vals in idol_values.items()}
    
    # 更新图的权重
    for name, effectiveness in idol_avg.items():
        source_name, target_name = name
        
        distance = float('inf')
        if effectiveness and effectiveness > 0:
            distance = 1 / effectiveness
        
        G.add_edge(source_name, target_name, weight=distance, effectiveness=effectiveness)
    return G

def get_dataset():
    G = build_collaboration_graph()
    
    vec_G = node2vec.Graph(G, is_directed=True, p=4, q=0.25)
    vec_G.preprocess_transition_probs()
    walks = vec_G.simulate_walks(num_walks=1, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    embedding_model = Word2Vec(walks, vector_size=512, window=10, min_count=0, sg=1, workers=8, epochs=5)
  
    data = []
    
    node_pairs =set()
    for u in G.nodes():
      for v in G.nodes():
        if u == v:
          continue
        
        if (u, v) in node_pairs or (v, u) in node_pairs:
          continue
        
        node_pairs.add((u, v))
    
    for u, v in node_pairs:
        sample = {
          "source": embedding_model.wv[u],
          "target": embedding_model.wv[v],
        }
        
        if G.has_edge(u, v) or G.has_edge(v, u):
          sample["label"] = 1
        else:
          sample["label"] = 0
        
        data.append(sample)
        
        
    positive_data = [sample for sample in data if sample["label"] == 1]
    negative_data = [sample for sample in data if sample["label"] == 0]
    
    num_of_positive = len(positive_data)
    num_of_negative = len(negative_data)
    
    # 隨機抽取負樣本
    negative_samples = random.sample(negative_data, num_of_positive)
    balanced_data = positive_data + negative_samples
    
    print(f"num_of_positive: {num_of_positive}, num_of_negative: {num_of_negative}")
    return MyDataset(balanced_data)

def get_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train():
  device = "mps"
  
  dataset = get_dataset()
  dataloader = get_dataloader(dataset, batch_size=256)
  model = node2vecClassifier.Node2VecClassifier().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()
  
  for epoch in range(10):
    total_loss = 0
    for batch in tqdm(dataloader):
      source = batch["source"].to(device)
      target = batch["target"].to(device)
      label = batch["label"].to(device)
      
      optimizer.zero_grad()
      output = model(source, target)
      # output = nn.functional.softmax(output, dim=1)
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    torch.save(model.state_dict(), f"{ROOT_DIR}/model/epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_node2vec_classifier.pth")
    print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")
    
if __name__ == "__main__":
  train()