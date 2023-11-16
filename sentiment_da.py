import os
import re

import newspaper
import pandas as pd
import base64
import functools

from pygooglenews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import statistics as sts
import translators as trans
import langdetect as ldet


# Ref: https://stackoverflow.com/a/59023463/

_ENCODED_URL_PREFIX = "https://news.google.com/rss/articles/"
_ENCODED_URL_RE = re.compile(fr"^{re.escape(_ENCODED_URL_PREFIX)}(?P<encoded_url>[^?]+)")
_DECODED_URL_RE = re.compile(rb'^\x08\x13".+?(?P<primary_url>http[^\xd2]+)\xd2\x01')

ANALYSIS_FOLDER_PATH = r'D:\CondaPy - Projects\Various\Stock prediction (ML)\Sentiment analysis'

USER_AGENT = \
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
scrap_config = newspaper.Config()
scrap_config.browser_user_agent = USER_AGENT
scrap_config.request_timeout = 10

GOOGLE_DATE_FORMAT = '%a, %d %b %Y %H:%M:%S %Z'

ANALYZER = SentimentIntensityAnalyzer()


@functools.lru_cache(2048)
def _decode_google_news_url(url: str) -> str:
    match = _ENCODED_URL_RE.match(url)
    encoded_text = match.groupdict()["encoded_url"]  # type: ignore
    encoded_text += "==="  # Fix incorrect padding. Ref: https://stackoverflow.com/a/49459036/
    decoded_text = base64.urlsafe_b64decode(encoded_text)

    match = _DECODED_URL_RE.match(decoded_text)
    primary_url = match.groupdict()["primary_url"]  # type: ignore
    primary_url = primary_url.decode()
    return primary_url


def decode_google_news_url(url: str) -> str:  # Not cached because not all Google News URLs are encoded.
    """Return Google News entry URLs after decoding their encoding as applicable."""
    return _decode_google_news_url(url) if url.startswith(_ENCODED_URL_PREFIX) else url


# Stock name followed by its type preferable, to enhance the results accuracy i.e. DAX Index, AAPL company
def prepare_articles(stock_name: str):
    found_news = get_news(stock_name)['entries']
    stock_name = stock_name.replace("'", "")
    articles = []
    downloaded, missed = 0, 0
    i = 1
    print(f"Analyzing {len(found_news)} articles...")
    for entry in found_news:
        print(f"\rRunning {i}/{len(found_news)}...", end='', flush=True)
        article_google_href = entry['links'][0]['href']
        decoded_url = decode_google_news_url(article_google_href)
        article = newspaper.Article(url=decoded_url, config=scrap_config)
        try:
            article.download()
            article.parse()
            article.text = trans.translate_text(
                article.text,
                translator='apertium',
                from_language=ldet.detect(article.text)
            )
            article.nlp()
            downloaded += 1
        except newspaper.article.ArticleException as e:
            missed += 1
            article.text = 'no access'
            article.summary = e

        articles.append({
            'title': entry['title'],
            'url': decoded_url,
            'native_lang': ldet.detect(article.text),
            'published': datetime.strptime(entry['published'], GOOGLE_DATE_FORMAT).date(),
            'summary': str(trans.translate_text(article.summary)),
            'sentiment': ANALYZER.polarity_scores(str(article.text))['compound']
        })
        i += 1

    print(f"Analyze finished with {round(missed / (missed + downloaded) * 100, 1)}% not analyzed feeds. ")
    articles_df = pd.DataFrame(articles).sort_values('published')
    articles_df = calculate_mean_sentiment(articles_df)
    print(f"Saving the analysis to Sentiment analysis/{stock_name}.csv")
    articles_df.to_csv(f'Sentiment analysis/{stock_name}.csv', sep=';')

    return articles_df, missed, missed + downloaded


def check_prev_analysis(stock_name: str):
    full_path = os.path.join(ANALYSIS_FOLDER_PATH, f"{stock_name}.csv")
    return os.path.exists(full_path)


def get_news(searched_prompt: 'str'):
    gn = GoogleNews()
    tcker_news = gn.search(f"${searched_prompt}")
    print(
        f"Fetching news for --------------------------- {searched_prompt} ----------------------------------------------")
    return tcker_news


def calculate_mean_sentiment(sorted_df: pd.DataFrame):
    unique_dates = tuple(sorted_df['published'].values)
    for date in unique_dates:
        all_articles_within_day = sorted_df[sorted_df['published'] == date]
        mean_sentiment = sts.mean(all_articles_within_day['sentiment'].values)
        sorted_df.loc[sorted_df['published'] == date, 'intraday_sentiment'] = mean_sentiment
    return sorted_df


if __name__ == "__main__":
    stock = "'WIG20'"

    prepare_articles(stock)
