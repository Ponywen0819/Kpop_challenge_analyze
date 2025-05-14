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
import node2vecClassifier
import pandas as pd
# 获取当前文件的目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
ROOT_DIR = os.path.dirname(CURRENT_DIR)

model = node2vecClassifier.Node2VecClassifier()

model.load_state_dict(torch.load(f"{ROOT_DIR}/model/epoch_9_20250506_111133_node2vec_classifier.pth"))
model.eval()
model.to("mps")

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
    
def check_should_be_ignored(video, cut_off_date, time_window):
    """檢查是否應該忽略影片"""
    
    video_date = datetime.fromtimestamp(video['timestamp'])
    if cut_off_date:
        if video_date >= cut_off_date or video_date < cut_off_date - timedelta(days=time_window):
            return True
    return False

def build_collaboration_graph(collaboration_data, time_window=30, cut_off_date=None):
    """建立合作网络图"""
    G = nx.DiGraph()


    # 建立節點
    for video in collaboration_data:
        source_member = video['source']
        target_member = video['target']
        G.add_node(f"{source_member[0]}_{source_member[1]}")
        G.add_node(f"{target_member[0]}_{target_member[1]}")
    
    # 建立邊
    for video in collaboration_data:
        # 如果時間窗口過期，則跳過
        if check_should_be_ignored(video, cut_off_date, time_window):
            continue

        source_member = video['source']
        target_member = video['target']

        G.add_edge(f"{source_member[0]}_{source_member[1]}", f"{target_member[0]}_{target_member[1]}")
    
    # 计算边的权重
    idol_values = defaultdict(list)
    for video in collaboration_data:
        # 如果時間窗口過期，則跳過
        if check_should_be_ignored(video, cut_off_date, time_window):
            continue
        
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

def main():
    # 加载已清洗的数据
    collaboration_data = load_collaboration_data()
    time_window = 180

    # 为每个视频生成推荐
    output_rows = []
    itter_data = [d for d in collaboration_data if d['timestamp'] >= datetime.fromisoformat("2025-01-01").timestamp()]
    
    for video in tqdm(itter_data):

        G = build_collaboration_graph(collaboration_data, time_window=time_window, cut_off_date=datetime.fromtimestamp(video['timestamp']))
        
        vec_G = node2vec.Graph(G, is_directed=True, p=1, q=1)
        vec_G.preprocess_transition_probs()
        walks = vec_G.simulate_walks(num_walks=1, walk_length=80)
        walks = [list(map(str, walk)) for walk in walks]
        embedding_model = Word2Vec(walks, vector_size=512, window=10, min_count=0, sg=1, workers=8, epochs=5)
  
        source_member = video['source']
        target_member = video['target']
        
        source_name = f"{source_member[0]}_{source_member[1]}"
        target_name = f"{target_member[0]}_{target_member[1]}"
        
        raw_data = []
        predictions = []
        node_list = list(G.nodes())
        source_embedding = torch.tensor(embedding_model.wv[source_name]).to("mps")
        for node in node_list:
            raw_data.append({
              "source": source_embedding,
              "target": torch.tensor(embedding_model.wv[node])
            })
            
        dataset = MyDataset(raw_data)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        offset = 0
        for batch in dataloader:
            source_embedding = batch["source"].to("mps")
            target_embedding = batch["target"].to("mps")
            model_output = model(source_embedding, target_embedding)
        
        
            for i in range(len(model_output)):
                predictions.append((node_list[i + offset], model_output[i][1]))
            offset += len(model_output)
      
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        output_rows.append({
          "date": datetime.fromtimestamp(video['timestamp']).strftime("%Y-%m-%d"),
          "initiator": source_name,
          "collaborator": target_name,
          "recommend_idol": ", ".join([name for name, _ in predictions]),
          "video_id": video.get("video_id", ""),
        })
    
    # 保存结果
    output_file = os.path.join(CURRENT_DIR, f"predictions_{time_window}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    df_output = pd.DataFrame(output_rows)
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()  



