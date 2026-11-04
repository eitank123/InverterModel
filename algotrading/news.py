"""News as a trading input: curated sources, a feed reader, headline scorers, and a
daily sentiment index that strategies can consume.

The hard parts of news-based trading are not the NLP:
  1. Timestamps. A headline dated "2015-03-10" may have been published after that day's
     close. Using it to trade the 10th is lookahead. Always trade the *next* session.
  2. Archives. You can only backtest sources with point-in-time historical archives.
     Today's RSS feed tells you nothing about 2008.
  3. Speed. Market-moving headlines are priced in within seconds by machines reading
     the wires. A retail strategy can only hope to capture slow drift, not the reaction.
  4. LLM lookahead. A language model trained on text up to year Y "knows" what
     happened after any headline before Y. Its scores on old headlines are contaminated;
     only headlines published after the model's training cutoff are a fair test.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

from .strategy import Strategy


@dataclass(frozen=True)
class NewsSource:
    name: str
    url: str          # machine-readable endpoint (RSS / API)
    kind: str         # "primary", "wire", "financial_media", "archive"
    why: str
    historical: bool  # point-in-time archive usable for backtests?


# Selection criteria: (1) machine-readable, (2) timestamped, (3) primary sources over
# commentary, (4) ideally a historical archive so the signal can be backtested.
SOURCES = [
    NewsSource("Federal Reserve press releases", "https://www.federalreserve.gov/feeds/press_all.xml",
               "primary", "FOMC rate decisions and statements move every asset class.", True),
    NewsSource("SEC EDGAR filings (8-K)", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom",
               "primary", "Company-material events straight from the source, with exact timestamps.", True),
    NewsSource("BLS economic releases", "https://www.bls.gov/feed/bls_latest.rss",
               "primary", "CPI and jobs reports: the scheduled macro surprises.", True),
    NewsSource("GDELT 2.0 DOC API", "https://api.gdeltproject.org/api/v2/doc/doc?query=economy&mode=artlist&format=json",
               "archive", "Global news since 2015 with tone scores: free and backtestable.", True),
    NewsSource("MarketWatch top stories", "https://feeds.content.dowjones.io/public/rss/mw_topstories",
               "financial_media", "Broad US market headlines (Dow Jones).", False),
    NewsSource("CNBC markets", "https://www.cnbc.com/id/100003114/device/rss/rss.html",
               "financial_media", "Fast US market coverage.", False),
    NewsSource("Yahoo Finance per-ticker", "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
               "financial_media", "Headlines for a specific ticker.", False),
]
# Paid but best-in-class: Reuters / Bloomberg / Dow Jones Newswires (machine-readable,
# millisecond timestamps, full archives), as used by funds. RavenPack sells pre-scored news.


def fetch_rss(url: str, timeout: float = 15.0) -> pd.DataFrame:
    """Read an RSS 2.0 or Atom feed into a DataFrame (published, title, link)."""
    req = urllib.request.Request(url, headers={"User-Agent": "algotrading-research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        root = ET.fromstring(resp.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    rows = []
    for it in root.iter("item"):
        date = it.findtext("pubDate")
        rows.append({"published": parsedate_to_datetime(date) if date else pd.NaT,
                     "title": (it.findtext("title") or "").strip(), "link": it.findtext("link")})
    for it in root.iterfind(".//a:entry", ns):
        link = it.find("a:link", ns)
        rows.append({"published": pd.to_datetime(it.findtext("a:updated", namespaces=ns)),
                     "title": (it.findtext("a:title", namespaces=ns) or "").strip(),
                     "link": link.get("href") if link is not None else None})
    return pd.DataFrame(rows, columns=["published", "title", "link"])


# --- Scorers -------------------------------------------------------------------------

class LexiconScorer:
    """Loughran-McDonald finance dictionary (``pip install pysentiment2``).

    Generic sentiment lists mislabel finance text ("liability", "tax", "cost" are not
    negative in a 10-K). LM was built from SEC filings for exactly this reason.
    score = (positive - negative) / (positive + negative), in [-1, 1]; 0 if no hits.
    """

    def __init__(self):
        import pysentiment2 as ps
        self._lm = ps.LM()

    def score(self, headlines: list[str]) -> np.ndarray:
        out = np.zeros(len(headlines))
        for i, h in enumerate(headlines):
            s = self._lm.get_score(self._lm.tokenize(h))
            n = s["Positive"] + s["Negative"]
            out[i] = (s["Positive"] - s["Negative"]) / n if n else 0.0
        return out


class ClaudeScorer:
    """Score headlines with Claude: -1 (very bad for US stocks) .. +1 (very good).

    Needs ``pip install anthropic`` and an API key (ANTHROPIC_API_KEY). Sends headlines
    in batches with a JSON schema so the reply is machine-parseable. Caveat: on
    headlines before the model's training cutoff the scores suffer lookahead bias.
    """

    SYSTEM = ("You are a financial news analyst. For each headline, estimate its likely "
              "impact on the broad US stock market over the next trading day, from -1 "
              "(very negative) to +1 (very positive); 0 if irrelevant. Judge only from the "
              "headline text, as if you were reading it on the day it was published.")

    def __init__(self, model: str = "claude-opus-5-5", batch_size: int = 50, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.batch_size = client, model, batch_size

    def _schema(self, n: int) -> dict:
        return {"type": "json_schema", "schema": {
            "type": "object",
            "properties": {"scores": {"type": "array", "items": {"type": "number"},
                                      "description": f"exactly {n} scores, in input order"}},
            "required": ["scores"], "additionalProperties": False}}

    def score(self, headlines: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(headlines), self.batch_size):
            batch = headlines[i:i + self.batch_size]
            numbered = "\n".join(f"{j + 1}. {h}" for j, h in enumerate(batch))
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=self.SYSTEM,
                output_config={"effort": "low", "format": self._schema(len(batch))},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": numbered}],
            )
            if response.stop_reason != "end_turn":
                raise RuntimeError(f"scoring stopped early: {response.stop_reason}")
            text = next(b.text for b in response.content if b.type == "text")
            scores = json.loads(text)["scores"]
            if len(scores) != len(batch):
                raise ValueError(f"expected {len(batch)} scores, got {len(scores)}")
            out.extend(np.clip(scores, -1, 1))
        return np.asarray(out, dtype=float)


# --- Daily index and strategy --------------------------------------------------------

def daily_sentiment(headlines: pd.DataFrame, scorer, date_col: str = "date",
                    text_col: str = "title") -> pd.Series:
    """Average headline score per calendar date."""
    scores = scorer.score(headlines[text_col].tolist())
    return pd.Series(scores, index=headlines.index).groupby(headlines[date_col]).mean()


class SentimentStrategy(Strategy):
    """Long when smoothed sentiment is above its trailing median, else ``low`` (cash/short).

    ``sentiment`` is indexed by the date the news was *published*. The backtester must
    run with ``execution_lag >= 1`` so day-t news is only traded from the next session.
    """

    def __init__(self, sentiment: pd.Series, smooth: int = 1, baseline: int = 250,
                 low: float = 0.0):
        self.sentiment, self.smooth, self.baseline, self.low = sentiment, smooth, baseline, low

    def positions(self, prices: pd.Series) -> pd.Series:
        s = self.sentiment.rolling(self.smooth).mean()
        thresh = s.rolling(self.baseline, min_periods=self.baseline // 2).median()
        sig = pd.Series(np.where(s > thresh, 1.0, self.low), index=s.index).where(thresh.notna())
        # Map news dates onto trading days: news on a weekend counts from the next bar.
        sig = sig.reindex(sig.index.union(prices.index)).ffill().reindex(prices.index)
        return sig.fillna(0.0)

    def __repr__(self) -> str:
        return f"SentimentStrategy(smooth={self.smooth}, baseline={self.baseline}, low={self.low})"


# --- Historical headline dataset -----------------------------------------------------

REDDIT_DJIA_URL = ("https://raw.githubusercontent.com/ShravanChintha/Stock-Market-prediction-"
                   "using-daily-news-headlines/master/Combined_News_DJIA.csv")


def load_reddit_headlines(cache_dir: str | None = None) -> pd.DataFrame:
    """Top-25 r/worldnews headlines per day, 2008-08 to 2016-07 (Kaggle "Daily News for
    Stock Market Prediction"). Long format: one row per headline (date, rank, title).

    Caveat: dates are UTC posting days, so some headlines appeared after the US close.
    """
    cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(cache_dir, "reddit_worldnews_2008_2016.csv")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        raw = pd.read_csv(REDDIT_DJIA_URL)
        raw.to_csv(path, index=False)
    raw = pd.read_csv(path, parse_dates=["Date"])
    long = raw.melt(id_vars=["Date", "Label"], value_vars=[f"Top{i}" for i in range(1, 26)],
                    var_name="rank", value_name="title").dropna(subset=["title"])
    # Strip the b'...' / b"..." wrappers left over from Python 2 byte strings.
    long["title"] = long["title"].astype(str).str.replace(r'^b["\'](.*)["\']$', r"\1", regex=True)
    long["rank"] = long["rank"].str[3:].astype(int)
    long = long.rename(columns={"Date": "date", "Label": "label"})
    return long.sort_values(["date", "rank"]).reset_index(drop=True)


# --- Real-time news monitor (Exp 28) ---------------------------------------------------

NEWS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "news")
GDELT_TONE_PATH = os.path.join(NEWS_DIR, "gdelt_tone_stock_market.csv")
HEADLINES_PATH = os.path.join(NEWS_DIR, "headlines.csv")
GDELT_TONE_URL = ("https://api.gdeltproject.org/api/v2/doc/doc?query=%22stock%20market%22%20sourcelang:english"
                  "&mode=timelinetone&format=json&timespan=3months")
LIVE_FEEDS = {"MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
              "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
              "Yahoo Finance (SPY)": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY"}


def update_gdelt_tone(path: str = GDELT_TONE_PATH, timeout: float = 60.0) -> pd.Series:
    """Append newly published days of GDELT's daily 'stock market' news tone to ``path``.
    Days already stored are never rewritten (point-in-time record). Offline -> stored data."""
    old = pd.read_csv(path, index_col=0, parse_dates=True)["tone"] if os.path.exists(path) else pd.Series(dtype=float)
    try:
        import time
        req = urllib.request.Request(GDELT_TONE_URL, headers={"User-Agent": "algotrading-research/0.1"})
        for attempt in range(4):                  # GDELT allows ~1 request per 5 s; 429 otherwise
            try:
                d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
                break
            except urllib.error.HTTPError as err:
                if err.code != 429 or attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))
        s = pd.DataFrame(d["timeline"][0]["data"])
        new = pd.Series(s["value"].to_numpy(), index=pd.to_datetime(s["date"].str[:8]), name="tone")
        today = pd.Timestamp.today().normalize()
        new = new[(new.index < today) & ~new.index.isin(old.index)]      # completed days only
        if len(new):
            old = pd.concat([old, new]).sort_index().rename("tone")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            old.rename_axis("date").to_csv(path)
    except Exception:
        pass
    return old


def tone_zscore(tone: pd.Series, smooth: int = 5, window: int = 250) -> pd.Series:
    """5-day average tone relative to its trailing 250-day mean, in standard deviations."""
    s = tone.rolling(smooth).mean()
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def collect_headlines(path: str = HEADLINES_PATH, feeds: dict | None = None, scorer=None) -> pd.DataFrame:
    """Fetch the live feeds, score each headline (Loughran-McDonald), append new ones to
    ``path`` (deduplicated by title) and return the whole archive. Feeds that fail are skipped."""
    feeds = feeds or LIVE_FEEDS
    old = pd.read_csv(path, parse_dates=["published", "fetched"]) if os.path.exists(path) else \
        pd.DataFrame(columns=["published", "fetched", "source", "title", "score", "link"])
    frames = []
    for name, url in feeds.items():
        try:
            df = fetch_rss(url)
            df["source"] = name
            frames.append(df)
        except Exception:
            continue
    if frames:
        new = pd.concat(frames, ignore_index=True)
        new = new[~new["title"].isin(old["title"]) & (new["title"] != "")].drop_duplicates("title")
        if len(new):
            scorer = scorer or LexiconScorer()
            new["score"] = scorer.score(new["title"].tolist())
            new["fetched"] = pd.Timestamp.now(tz="UTC")
            new["published"] = pd.to_datetime(new["published"], utc=True, errors="coerce")
            old = pd.concat([old, new[old.columns]], ignore_index=True)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            old.to_csv(path, index=False)
    return old
