import os

#  Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SCRAPED_JSON = os.path.join(OUTPUT_DIR, "01_scraped_news.json")

#  Scraping

# BBC News RSS feeds
BBC_FEEDS: dict[str, str] = {
    "business":      "http://feeds.bbci.co.uk/news/business/rss.xml",
    "politics":      "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "technology":    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "science":       "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "entertainment": "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "health":        "http://feeds.bbci.co.uk/news/health/rss.xml",
    "world":         "http://feeds.bbci.co.uk/news/world/rss.xml",
}

REQUEST_TIMEOUT = 15
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NLP-Lab/1.0",
}

#  NLP / ML

TFIDF_MAX_FEATURES = 5000
TFIDF_MAX_DF = 0.85
TEST_SIZE = 0.25
RANDOM_STATE = 42

#  Plots

FIGURE_DPI = 150
TOP_N_WORDS = 30
DISPERSION_TARGETS_N = 10
BIGRAM_TOP_N = 40
