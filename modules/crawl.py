import requests
import json
import shutil
import openai
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

openai.api_key = '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'

def give_idx(news):
  lst = [i for i in range(len(news)) if len(news[i]) > 2]
  return lst[0], lst[-1]

def crawl():
    ## 날씨기사
    url_weather = 'https://www.ytn.co.kr/weather/list_weather.php'
    resp_weather = requests.get(url_weather)
    soup_weather = BeautifulSoup(resp_weather.content, 'lxml')

    ## 과학기사
    url_science = 'https://www.ytn.co.kr/news/list.php?mcd=0105'
    resp_science = requests.get(url_science)
    soup_science = BeautifulSoup(resp_science.content, 'lxml')

    ## 문화기사
    url_culture = 'https://www.ytn.co.kr/news/list.php?mcd=0106'
    resp_culture = requests.get(url_culture)
    soup_culture = BeautifulSoup(resp_culture.content, 'lxml')

    ## 국제기사
    url_international = 'https://www.ytn.co.kr/news/list.php?mcd=0104'
    resp_international = requests.get(url_international)
    soup_international = BeautifulSoup(resp_international.content, 'lxml')

    news_tags_weather = soup_weather.select('ul a span.til')
    news_info_weather = soup_weather.select('ul li a')

    news_tags_science = soup_science.select('ul a span.til')
    news_info_science = soup_science.select('ul li a')

    news_tags_culture = soup_culture.select('ul a span.til')
    news_info_culture = soup_culture.select('ul li a')

    news_tags_international = soup_international.select('ul a span.til')
    news_info_international = soup_international.select('ul li a')

    news_titles_weather = [list(tags)[0] for tags in news_tags_weather]

    news_titles_science = [list(tags)[0] for tags in news_tags_science]

    news_titles_culture = [list(tags)[0] for tags in news_tags_culture]

    news_titles_international = [list(tags)[0] for tags in news_tags_international]

    start_w, end_w = give_idx(news_info_weather)
    start_s, end_s = give_idx(news_info_science)
    start_c, end_c = give_idx(news_info_culture)
    start_i, end_i = give_idx(news_info_international)

    news_info_weather = news_info_weather[start_w:end_w]
    news_info_science = news_info_science[start_s:end_s]
    news_info_culture = news_info_culture[start_c:end_c]
    news_info_international = news_info_international[start_i:end_i]

    urls_weather = [news_info_weather[i]['href'] for i in range(len(news_info_weather))]
    urls_science = [news_info_science[i]['href'] for i in range(len(news_info_science))]
    urls_culture = [news_info_culture[i]['href'] for i in range(len(news_info_culture))]
    urls_international = [news_info_international[i]['href'] for i in range(len(news_info_international))]

    articles_weather = [BeautifulSoup(requests.get(item).content, 'lxml').select('div.content div.box span') for item in
                        urls_weather]
    articles_science = [BeautifulSoup(requests.get(item).content, 'lxml').select('div.content div.box span') for item in
                        urls_science]
    articles_culture = [BeautifulSoup(requests.get(item).content, 'lxml').select('div.content div.box span') for item in
                        urls_culture]
    articles_international = [BeautifulSoup(requests.get(item).content, 'lxml').select('div.content div.box span') for
                              item in urls_international]

    summaries_weather = []
    # for main in articles_weather[::]: #양이 많으면 token 수 제한으로 오류
    weather = articles_weather[0]
    try:
        key = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": "I want you to return 4 keywords in English from the article below. Here is the article:" + str(
                     weather)}
            ]
        )
        summaries_weather.append(key['choices'][0]['message']['content'])
    except:
        pass

    summaries_science = []
    science = articles_science[0]
    try:
        key = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": "I want you to return 4 keywords in English from the article below. Here is the article:" + str(
                     science)}
            ]
        )
        summaries_science.append(key['choices'][0]['message']['content'])
    except:
        pass

    summaries_culture = []
    culture = articles_culture[0]
    try:
        key = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": "I want you to return 4 keywords in English from the article below. Here is the article:" + str(
                     culture)}
            ]
        )
        summaries_culture.append(key['choices'][0]['message']['content'])
    except:
        pass

    summaries_international = []
    international = articles_international[0]
    try:
        key = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": "I want you to return 4 keywords in English from the article below. Here is the article:" + str(
                     international)}
            ]
        )
        summaries_international.append(key['choices'][0]['message']['content'])
    except:
        pass

    return summaries_weather, summaries_science, summaries_culture, summaries_international
