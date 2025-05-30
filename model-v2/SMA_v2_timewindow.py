import json
import pandas as pd
from pyvis.network import Network
from collections import Counter, defaultdict
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
import os
import pandas as pd
from topkevaluation import calculate_top_k_metrics

# 获取当前文件的目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
ROOT_DIR = os.path.dirname(CURRENT_DIR)

def load_collaboration_data():
    """加载已清洗的合作视频数据"""
    df = pd.read_csv("/home/bl515-ml/Documents/shaio_jie/sma/Kpop_challenge_analyze/data/collaboration_videos.csv")
    df['timestamp'] = df['timestamp'].astype(float)
    
    # 設置分割時間戳（2025-01-01 的 Unix 時間戳）
    split_timestamp = datetime(2025, 1, 1).timestamp()
    
    # train_df = df[df['timestamp'] >= split_timestamp]
    
    # 修改过滤逻辑
    df = df[df['source'].str.split('_').str[0] != df['target'].str.split('_').str[0]]
    return df
    

# region 建立合作網絡圖

def check_should_be_ignored(video, cut_off_date, time_window):
    """檢查是否應該忽略影片"""
    
    video_date = datetime.fromtimestamp(video['timestamp'])
    if cut_off_date:
        if video_date >= cut_off_date or video_date < cut_off_date - timedelta(days=time_window):
            return True
    return False

def get_all_nodes():
    """獲取所有節點"""
    
    df = pd.read_csv("/home/bl515-ml/Documents/shaio_jie/sma/Kpop_challenge_analyze/data/node_list.csv")
    return df['node'].tolist()

def build_collaboration_graph(collaboration_data, time_window=30, cut_off_date=None):
    """建立合作网络图"""
    G = nx.DiGraph()
    

    # 建立節點
    for node in get_all_nodes():
        G.add_node(node)
    # 建立邊
    for _, row in collaboration_data.iterrows():
        # 如果時間窗口過期，則跳過
        if check_should_be_ignored(row, cut_off_date, time_window):
            continue

        source_member = row['source']
        target_member = row['target']

        G.add_edge(source_member, target_member)
    
    # 计算边的权重
    idol_values = defaultdict(list)
    for _, row in collaboration_data.iterrows():
        # 如果時間窗口過期，則跳過
        if check_should_be_ignored(row, cut_off_date, time_window):
            continue
        
        source_member = row['source']
        target_member = row['target']
        effectiveness = row['views'] * 0.5 + row['likes'] * 0.3 + row['comments'] * 0.2
        
        combination = (source_member, target_member)
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

# endregion

def directed_jaccard(G, node):
    if node not in G:
        print(f"節點 {node} 不在圖中。")
        return []  
    neighbors = set(G.successors(node))  # 出邊
    scores = []
    for other in G.nodes():
        if other == node:
            continue
        other_neighbors = set(G.successors(other))
        intersection = neighbors & other_neighbors
        union = neighbors | other_neighbors
        if union:
            score = len(intersection) / len(union)
            scores.append((node, other, score))
    return scores

def directed_adamic_adar(G, node):
    neighbors = set(G.successors(node))  # 出邊
    scores = []
    for other in G.nodes():
        if other == node:
            continue
        other_neighbors = set(G.successors(other))
        common = neighbors & other_neighbors
        score = 0.0
        for z in common:
            degree = len(set(G.successors(z))) + len(set(G.predecessors(z)))
            if degree > 1:
                score += 1 / np.log(degree)
        scores.append((node, other, score))
    return scores

def directed_shortest_path(G, source):
    lengths = {}
    try:
        lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
    except Exception as e:
        print(f"Error computing shortest paths for {source}: {e}")
    scores = [(source, target, length) for target, length in lengths.items() if target != source]
    return scores

def personalized_pagerank(G, source, alpha=0.85):
    try:
        personalization = {node: 0 for node in G.nodes()}
        personalization[source] = 1  # Center the PageRank on the source node
        return nx.pagerank(G, alpha=alpha, personalization=personalization, weight='effectiveness')
    except ZeroDivisionError:
        print(f"ZeroDivisionError occurred for source node {source}")
        return {}

def find_score(name, scores, reverse=True):
    cop = defaultdict(float)
    for u, v, score in scores:
        if name == u:    # 只看 name 出發的邊
            cop[v] = score
    sorted_cop = sorted(cop.items(), key=lambda x: x[1], reverse=reverse)
    return sorted_cop

def recommend(G, name, threshold=1):
    # 計算這個 idol 的 link prediction 分數
    if name not in G:
        return []
    pagerank_scores_all = personalized_pagerank(G, name, alpha=0.85)
    jaccard_scores_all = directed_jaccard(G, name)
    shortest_path_all = directed_shortest_path(G, name)

    sorted_pagerank = sorted(pagerank_scores_all.items(), key=lambda x: x[1], reverse=True)
    sorted_jaccard = find_score(name, jaccard_scores_all, reverse=True)
    sorted_shortest = find_score(name, shortest_path_all, reverse=False)

    # 取 Top % 百分比
    top_n_pagerank = max(1, int(len(sorted_pagerank) * threshold))
    top_n_jaccard = max(1, int(len(sorted_jaccard) * threshold))
    top_n_shortest = max(1, int(len(sorted_shortest) * threshold))

    top_pagerank = {idol for idol, _ in sorted_pagerank[:top_n_pagerank]}
    top_jaccard = {idol for idol, _ in sorted_jaccard[:top_n_jaccard]}
    top_shortest = {idol for idol, _ in sorted_shortest[:top_n_shortest]}

    # 三個都符合的候選名單
    candidates = top_pagerank & top_jaccard & top_shortest

    scored_candidates = []
    for idol_candidate in candidates:
        # 用 link prediction 分數推估效益
        pagerank = dict(sorted_pagerank).get(idol_candidate, 0)
        jaccard = dict(sorted_jaccard).get(idol_candidate, 0)
        shortest = dict(sorted_shortest).get(idol_candidate, 0)

        if shortest > 0:
            shortest_score = 1 / shortest
        else:
            shortest_score = 0
        effectiveness = (100 * pagerank + 100 * jaccard + shortest_score) / 3  # 平均
        scored_candidates.append((idol_candidate, effectiveness))
    # 效益高的排前面
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates

def main():
    # 加载已清洗的数据
    collaboration_data = load_collaboration_data()
    time_window = 60

    # 为每个视频生成推荐
    output_rows = []
    for _, row in collaboration_data.iterrows():
        video_date = datetime.fromtimestamp(row['timestamp'])
        if video_date < datetime(2025, 1, 1):
            continue
        
        
        G = build_collaboration_graph(collaboration_data, time_window=time_window, cut_off_date=video_date)
        source_name = row['source']
        target_name = row['target']
        
        recommendations = recommend(G, source_name, threshold=1)
        
        if recommendations:
            recommend_idol = []
            for recommend_team in recommendations:
                recommend_idol.append(recommend_team[0])
            
            output_rows.append({
                "date": datetime.fromtimestamp(row['timestamp']).strftime("%Y-%m-%d"),
                "initiator": source_name,
                "collaborator": target_name,
                "recommend_idol": ", ".join(recommend_idol),
                "video_id": row.get("video_id", ""),
            })
    
    # 保存结果
    
    output_file = os.path.join(CURRENT_DIR, f"predictions_{time_window}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    df_output = pd.DataFrame(output_rows)
    
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')

    k=10
    accuracy, mrr = calculate_top_k_metrics(output_file, k=k)
            # 打印结果
    print(f"Top-{k} 评估指标:")
    print(f"Top-{k} 准确率 (Top-{k} Accuracy): {accuracy:.4f}")
    print(f"MRR (MRR): {mrr:.4f}")
if __name__ == "__main__":
    main()