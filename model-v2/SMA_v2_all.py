import json
import pandas as pd
from pyvis.network import Network
from collections import Counter, defaultdict
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 載入資料
with open("./v2-kpop-challenge-shorts.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# 載入 idol 與 group 清單
idol_name = pd.read_excel("./K-POP藝人清單.xlsx", sheet_name="idols")
group_name = pd.read_excel("./K-POP藝人清單.xlsx", sheet_name="groups")

# 建立團體與成員對照表
group_kor_to_eng = {}
group_to_idols = defaultdict(list)

# 將韓文團體名與對應的英文團體名及成員加入對照表
for _, row in group_name.iterrows():
    group_kor = row["group (korean)"]
    group_eng = row["group (english)"].upper()
    group_kor_to_eng[group_kor] = group_eng
    group_kor_to_eng[group_eng] = group_eng

# 建立每個團體的成員名單，包含韓文與英文名稱
for _, row in idol_name.iterrows():
    group_eng = row["group (english)"].upper()
    idol_eng = row["name (english)"].upper()
    idol_kor = row["name (korean)"]
    group_to_idols[group_eng].append((idol_eng, idol_kor))

# 解析影片中的 hashtags 並判斷發起人與合作人
idol_cop = []
for group in data:
    initiate_group_eng = group.upper()
    for video in data[group]["shorts"]:
        tags = [tag[1:] if tag.startswith("#") else tag for tag in video["hashtags"]]
        tags += video.get("title", "").split()
        initiators = set()
        collaborators = set()
        appeared_group = set()
        # 判斷出現的團體
        for tag in tags:
            tag_upper = tag.upper()
            if tag_upper in group_kor_to_eng:
                group_eng = group_kor_to_eng[tag_upper]
                appeared_group.add(group_eng)
        for grp in appeared_group:
            for tag in tags:
                tag_upper = tag.upper()          
                for idol_eng, idol_kor in group_to_idols[grp]:
                    full_name = f"{grp}_{idol_eng}"
                    if idol_eng == tag_upper or idol_kor == tag:
                        if grp == initiate_group_eng:
                            initiators.add(full_name)
                        else:
                            collaborators.add(full_name)
        if initiators:
            idols = (initiators, collaborators)
            video["idols"] = idols
            idol_cop.append(idols)


# 建立有向圖
G = nx.DiGraph()
edge_counter = Counter()
for initiators, collaborators in idol_cop:
    for init in initiators:
        if collaborators:
            for collab in collaborators:
                G.add_edge(init, collab)  # 一個 initiator 配一個 collaborator
                edge_counter[(init, collab)] += 1
        else:
            if len(initiators) == 2:  # 同團只考慮人數為二
                for init_2 in initiators:
                    if init != init_2:
                        G.add_edge(init, init_2)
                        edge_counter[(init, init_2)] += 1
                        

# pyvis 顯示
net = Network(notebook=True, directed=True)
net.toggle_physics(True)
net.set_options(""" 
{
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -100,
      "centralGravity": 0.01,
      "springLength": 150,
      "springConstant": 0.02,
      "damping": 0.8,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "stabilization": {
      "iterations": 150,
      "updateInterval": 25
    }
  }
}
""")
# 加入節點
for node in G.nodes():
    net.add_node(node, label=node, font={'size': 12, 'vadjust': -40})
# 加入邊
for (u, v), count in edge_counter.items():
    net.add_edge(u, v, label=str(count), color='#222222', font={'size': 12}, width=2)
# 輸出圖
net.show("collab_times.html")



video_infos = []  # 把每支影片的資料跟 group 綁一起
for group in data:
    for video in data[group]["shorts"]:
        video_value = {
            'group': group.upper(),  # 加上 group 資訊
            'views': video['views'],
            'likes': video['likes'],
            'comments': video['comments'],
            'video': video,
        }
        video_infos.append(video_value)

group_videos = defaultdict(list)
for info in video_infos:
    group_videos[info['group']].append(info)

# 定義 percentile rank 函數
def percentile_ranks(values):
    values = np.array(values)
    return [np.sum(values <= x) / len(values) * 100 for x in values]

# 計算每個 group 的 pr 值
for group, videos in group_videos.items():
    views = [v['views'] for v in videos]
    likes = [v['likes'] for v in videos]
    comments = [v['comments'] for v in videos]

    views_ranks = percentile_ranks(views)
    likes_ranks = percentile_ranks(likes)
    comments_ranks = percentile_ranks(comments)

    for idx, video in enumerate(videos):
        f_value = 0.5 * views_ranks[idx] + 0.3 * likes_ranks[idx] + 0.2 * comments_ranks[idx]
        video['final_value'] = f_value  # 新增 final_value

# 重新分配每支影片的效益到 idol 組合
idol_values = defaultdict(list)

for info in video_infos:
    video = info['video']
    if "idols" in video:
        initiators, collaborators = video["idols"]
        idol_teams = []
        for init in initiators:
            if collaborators:
                for collab in collaborators:
                    idol_team = (init, collab)
                    idol_teams.append(idol_team)
            else:
                if len(initiators) == 2:
                    for init_2 in initiators:
                        if init != init_2:
                            idol_team = (init, init_2)
                            idol_teams.append(idol_team)
        if idol_teams:
            for idol_team in idol_teams:
                idol_values[idol_team].append(info['final_value'])

# 計算每對idol組合的平均效益
idol_avg = {name: sum(vals) / len(vals) for name, vals in idol_values.items()}


G = nx.DiGraph()
for (u, v) in edge_counter:
    effectiveness = idol_avg.get((u, v))  # get() 方法會自動回傳 None
    if effectiveness and effectiveness > 0:
        distance = 1 / effectiveness
        G.add_edge(u, v, weight=1/effectiveness, effectiveness=effectiveness)

net = Network(notebook=True, directed=True)
# 讓圖動起來(靜態會很亂)
net.toggle_physics(True)
net.set_options("""
{
  "physics": {
    "enabled": true,
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -100,
      "centralGravity": 0.01,
      "springLength": 150,
      "springConstant": 0.02,
      "damping": 0.8,
      "avoidOverlap": 1
    },
    "minVelocity": 0.75,
    "stabilization": {
      "iterations": 150,
      "updateInterval": 25
    }
  }
}
""")
# 加入節點
for node in G.nodes():
    net.add_node(
        node,
        label=node,
        font={'size': 12, 'vadjust': -40}
    )
# 加入邊
for team, value in idol_avg.items():
    u = team[0]
    v = team[1]
    net.add_edge(
        u, v,
        label=str(round(value, 2)),  # 顯示合作效益
        color='#222222',
        font={'size': 30},
        title=f"{u} - {v} link has value {round(value, 2)}",
        width=2
    )
net.show("collab_value.html")



# 有向版 Jaccard coefficient
def directed_jaccard(G, node):
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

# 有向版 Adamic-Adar
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

# 最短路徑
def directed_shortest_path(G, source):
    lengths = {}
    try:
        lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
    except Exception as e:
        print(f"Error computing shortest paths for {source}: {e}")
    scores = [(source, target, length) for target, length in lengths.items() if target != source]
    return scores

def personalized_pagerank(G, source, alpha=0.85):
    personalization = {node: 0 for node in G.nodes()}
    personalization[source] = 1  # 以 source 為中心
    return nx.pagerank(G, alpha=alpha, personalization=personalization, weight='effectiveness')

def find_score(name, scores, reverse=True):
    cop = defaultdict(float)
    for u, v, score in scores:
        if name == u:    # 只看 name 出發的邊
            cop[v] = score
    sorted_cop = sorted(cop.items(), key=lambda x: x[1], reverse=reverse)
    return sorted_cop


def recommend(name, threshold=1):
    # 計算這個 idol 的 link prediction 分數
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

# 測試推薦功能
print(recommend('AESPA_WINTER', threshold=1))
