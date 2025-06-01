import feedparser
from dateutil import parser
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import html
import os
import json

class RSSReader:
    def __init__(self, keywords=None, cache_file='data/rss_cache.json'):
        self.keywords = [kw.lower() for kw in keywords] if keywords else []
        self.feeds = {
            'vnexpress': 'https://vnexpress.net/rss/suc-khoe.rss',
            'thanhnien': 'https://thanhnien.vn/rss/suc-khoe.rss',
            'suckhoedoisong': 'https://suckhoedoisong.vn/y-te.rss',
            'sggp': 'https://www.sggp.org.vn/rss/ytesuckhoe-212.rss', 
        }
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.last_fetched = None

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except:
                    return []
        return []

    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def clean_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text().strip()
        img = soup.find('img')
        img_url = img['src'] if img and 'src' in img.attrs else ''
        return text, img_url

    def _fetch_custom_rss(self, url, source):
        try:
            res = requests.get(url, timeout=10)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'xml')
            items = soup.find_all('item')
        except Exception as e:
            print(f"Error fetching {source}: {e}")
            return []

        results = []
        for item in items:
            title = html.unescape(item.title.text.strip())
            title_lower = title.lower()
            if self.keywords and not any(kw in title_lower for kw in self.keywords):
                continue

            description_html = item.description.text.strip()
            description, extracted_img = self.clean_html(html.unescape(description_html))

            pub_date = item.pubDate.text.strip()
            try:
                published_date = parser.parse(pub_date).strftime('%d/%m/%Y %H:%M')
            except:
                published_date = ''

            image = ''
            enclosure = item.find('enclosure')
            if enclosure and enclosure.has_attr('url'):
                image = enclosure['url']
            elif extracted_img:
                image = extracted_img

            results.append({
                'title': title,
                'description': description,
                'link': item.link.text.strip(),
                'published': published_date,
                'image': image,
                'source': source
            })
        return results

    def fetch_and_cache(self):
        results = []
        for source, url in self.feeds.items():
            if source == 'suckhoedoisong':
                results.extend(self._fetch_custom_rss(url, source))
                continue

            try:
                response = requests.get(url, timeout=10)
                response.encoding = 'utf-8'
                feed = feedparser.parse(response.text)
            except Exception as e:
                print(f"Error fetching {source}: {e}")
                continue

            for entry in feed.entries:
                raw_title = entry.title
                title = html.unescape(raw_title)
                title_lower = title.lower()

                if self.keywords and not any(kw in title_lower for kw in self.keywords):
                    continue

                published = entry.get('published', '') or entry.get('updated', '')
                try:
                    dt = parser.parse(published)
                    published_date = dt.strftime('%d/%m/%Y %H:%M')
                except Exception:
                    published_date = ''

                raw_description = entry.get('summary', entry.get('description', ''))
                description, extracted_img = self.clean_html(raw_description)

                image = ''
                if 'media_content' in entry:
                    image = entry.media_content[0].get('url', '')
                elif not image and extracted_img:
                    image = extracted_img

                results.append({
                    'title': title,
                    'link': entry.link,
                    'description': description,
                    'published': published_date,
                    'source': source,
                    'image': image
                })

        # Tích lũy cache: tránh trùng link
        existing_links = {item['link'] for item in self.cache}
        new_items = [item for item in results if item['link'] not in existing_links]

        if new_items:
            self.cache.extend(new_items)
            self.cache = sorted(self.cache, key=lambda x: x['published'], reverse=True)
            self._save_cache()

        self.last_fetched = datetime.utcnow()

    def get_news(self, source='all'):
        if not self.last_fetched or datetime.utcnow() - self.last_fetched > timedelta(hours=1):
            self.fetch_and_cache()

        if source != 'all':
            return [news for news in self.cache if news['source'] == source]

        return self.cache
