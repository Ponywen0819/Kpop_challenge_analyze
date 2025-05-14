import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import json
from urllib.parse import quote
import re
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import os

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)

class ArtistCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_wikipedia_content(self, artist_name):
        """從維基百科獲取藝人介紹"""
        try:
            # 構建搜索 URL，添加 K-POP 關鍵詞以提高準確性
            search_query = f"{artist_name} K-POP"
            search_url = f"https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(search_query)}&format=json"
            response = self.session.get(search_url)
            data = response.json()
            
            if not data['query']['search']:
                # 如果沒有找到結果，嘗試只搜索藝人名字
                search_url = f"https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(artist_name)}&format=json"
                response = self.session.get(search_url)
                data = response.json()
                
                if not data['query']['search']:
                    return None
            
            # 獲取第一個搜索結果的頁面 ID
            page_id = data['query']['search'][0]['pageid']
            
            # 獲取頁面內容
            content_url = f"https://zh.wikipedia.org/w/api.php?action=query&prop=extracts&pageids={page_id}&format=json"
            response = self.session.get(content_url)
            data = response.json()
            
            # 提取文本內容
            content = data['query']['pages'][str(page_id)]['extract']
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            
            # 清理文本
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        except Exception as e:
            logging.error(f"維基百科爬取錯誤 ({artist_name}): {str(e)}")
            return None
            
    def get_kpopwiki_content(self, artist_name):
        """從 K-POP Wiki 獲取藝人介紹"""
        try:
            # 構建搜索 URL
            search_url = f"https://kpop.fandom.com/wiki/Special:Search?query={quote(artist_name)}"
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 找到第一個搜索結果的鏈接
            result = soup.find('a', {'class': 'unified-search__result__title'})
            if not result:
                # 如果沒有找到結果，嘗試添加團體名稱
                group_name = self.get_group_name(artist_name)
                if group_name:
                    search_url = f"https://kpop.fandom.com/wiki/Special:Search?query={quote(f'{artist_name} {group_name}')}"
                    response = self.session.get(search_url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    result = soup.find('a', {'class': 'unified-search__result__title'})
                
                if not result:
                    return None
                
            # 訪問藝人頁面
            artist_url = result['href']
            response = self.session.get(artist_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取介紹部分
            content = soup.find('div', {'class': 'mw-parser-output'})
            if not content:
                return None
                
            # 提取文本
            text = content.get_text()
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        except Exception as e:
            logging.error(f"K-POP Wiki 爬取錯誤 ({artist_name}): {str(e)}")
            return None
            
    def get_group_name(self, artist_name):
        """從 CSV 文件中獲取藝人所在的團體名稱"""
        try:
            df = pd.read_csv('data/K-POP藝人清單.csv')
            artist_row = df[df['name (english)'] == artist_name]
            if not artist_row.empty:
                return artist_row['group (english)'].iloc[0]
            return None
        except Exception as e:
            logging.error(f"獲取團體名稱錯誤 ({artist_name}): {str(e)}")
            return None
            
    def get_artist_info(self, artist_name):
        """獲取藝人的所有可用信息"""
        info = {
            'wikipedia': None,
            'kpopwiki': None
        }
        
        # 從維基百科獲取信息
        wiki_content = self.get_wikipedia_content(artist_name)
        if wiki_content:
            info['wikipedia'] = wiki_content
            
        # 從 K-POP Wiki 獲取信息
        kpop_content = self.get_kpopwiki_content(artist_name)
        if kpop_content:
            info['kpopwiki'] = kpop_content
            
        # 合併所有文本
        all_text = ' '.join([text for text in info.values() if text])
        return all_text if all_text else None
        
    def crawl_artists(self, artist_list, output_file='artist_texts.json'):
        """爬取多個藝人的信息"""
        results = {}
        
        for artist in tqdm(artist_list, desc="爬取藝人信息"):
            try:
                # 添加隨機延遲
                time.sleep(random.uniform(1, 3))
                
                # 獲取藝人信息
                info = self.get_artist_info(artist)
                if info:
                    results[artist] = info
                    logging.info(f"成功爬取 {artist} 的信息")
                else:
                    logging.warning(f"未找到 {artist} 的信息")
                    
            except Exception as e:
                logging.error(f"爬取 {artist} 時發生錯誤: {str(e)}")
                continue
                
        # 保存結果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        return results

def main():
    # 讀取藝人列表
    df = pd.read_csv('data/K-POP藝人清單.csv')
    artist_list = df['name (english)'].unique().tolist()
    
    # 創建爬蟲實例
    crawler = ArtistCrawler()
    
    # 開始爬取
    results = crawler.crawl_artists(artist_list)
    
    # 輸出統計信息
    success_count = len(results)
    total_count = len(artist_list)
    logging.info(f"爬取完成！成功爬取 {success_count}/{total_count} 個藝人的信息")

if __name__ == "__main__":
    main() 