import json
import pandas as pd
import numpy as np
from datetime import datetime

idol_list = pd.read_csv("./data/K-POP藝人清單.csv")
with open("./data/v2-kpop-challenge-shorts.json", 'r', encoding='utf-8') as f:
    youtube_data = json.load(f)    

youtube_data = { f.upper(): youtube_data[f] for f in youtube_data }
group_name_list = set([ f.upper() for f in idol_list["group (english)"].tolist()])

group_member_dict = {}

for _, idol in idol_list.iterrows():
    group_belong_to = idol["group (english)"]

    # 為了方便查找，將團體名稱轉換為小寫
    upper_group_name = group_belong_to.upper()

    # 發現 HashTag 有時會出現只有韓文名稱的狀況，因此需要將韓文名稱也加入查找表
    upper_english_name = idol["name (english)"].upper()
    upper_korean_name = idol['name (korean)'].upper()

    if upper_group_name not in group_member_dict:
        group_member_dict[upper_group_name] = []

    group_member_dict[upper_group_name].append(upper_english_name + "," + upper_korean_name)


# 抽取影片中的團體
def getGroupsInVideo(tags):
    groups_in_video = []

    for tag in tags:
        if tag[0] == '#' and (tag[1:].upper() in group_name_list):
            groups_in_video.append(tag[1:].upper())

    return groups_in_video

# 使用團體成員列表查找影片中的合作藝人
def findGroupMember(str, group_members):
    for member in group_members:
        name_split = member.split(",")
        if len(name_split) != 2:
            print(name_split)
            assert False

        english_name,korean_name = name_split
        if str == english_name or str == korean_name:
            return english_name
    return None

# 抽取影片中的合作藝人
def getMembersInVideo(tags, group_list, group_member_dict):
    members_in_video = set()
    for tag in tags:
        for group in group_list:
            member = findGroupMember(tag[1:].upper(), group_member_dict[group])
            if member is not None:
                members_in_video.add((group,member))
    return list(members_in_video)

# 定義 percentile rank 函數
def percentile_ranks(values):
    values = np.array(values)
    return [np.sum(values <= x) / len(values) * 100 for x in values]

def get_collaboration_videos(source_group, group_data, group_member_dict):
    collaboration_videos = []
    videos = sorted(group_data['shorts'], key=lambda x: datetime.strptime(x['upload_time'], "%Y-%m-%d %H:%M:%S"))

    views = [v['views'] for v in videos]
    likes = [v['likes'] for v in videos]
    comments = [v['comments'] for v in videos]

    views_ranks = percentile_ranks(views)
    likes_ranks = percentile_ranks(likes)
    comments_ranks = percentile_ranks(comments)

        
    for i, video in enumerate(videos):
        print("processing .... ",video['title'], "from ", source_group)
        tags = video['hashtags']
        groups_in_video = getGroupsInVideo(tags)
        
        members_in_video = getMembersInVideo(tags, groups_in_video, group_member_dict)
            
        is_collaboration = len(members_in_video) < 2
        if is_collaboration:
            print(f"This video is not a collaboration video")
            continue
            
        if len(members_in_video) < len(groups_in_video):
            print("Can't find all members in video")
            continue

        source_members = [member for member in members_in_video if member[0] == source_group ]
        target_members = [member for member in members_in_video if member[0] != source_group ]


        
        payload = {
            "timestamp": datetime.strptime(video['upload_time'], "%Y-%m-%d %H:%M:%S").timestamp(),
            "views": views_ranks[i],
            "likes": likes_ranks[i],
            "comments": comments_ranks[i],
            "video_id": video['video_id']
        }

        if len(target_members) == 0:
            for source in source_members:
                for target in source_members:
                    if source == target:
                        continue
                    collaboration_videos.append({
                        "source":f"{source[0]}_{source[1]}",
                        "target":f"{target[0]}_{target[1]}",
                        **payload
                    })
        else:
            for source in source_members:
                for target in target_members:
                    collaboration_videos.append({
                        "source":f"{source[0]}_{source[1]}",
                        "target":f"{target[0]}_{target[1]}",
                        **payload
                    })

    print(len(collaboration_videos) / len(group_data['shorts']) * 100, "%")
    return collaboration_videos

collaboration_videos = []
for group in group_name_list:
    if group not in youtube_data:
        print(f"{group} not in data")
        continue
    
    single_group_videos = get_collaboration_videos(group, youtube_data[group], group_member_dict)
    collaboration_videos.extend(single_group_videos)

collaboration_videos = sorted(collaboration_videos, key=lambda x: x['timestamp'])

df = pd.DataFrame(collaboration_videos)

df.to_csv('data/collaboration_videos.csv', index=False, encoding='utf-8')
len(collaboration_videos)
