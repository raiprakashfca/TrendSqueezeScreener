import io
import requests
import pandas as pd
import streamlit as st

# Official NIFTY 50 constituents CSV from NSE archives
# This is the same "Download List of Nifty 50 stocks (.csv)"
# you see on the Nifty 50 index page.
NIFTY50_CSV_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"


def fetch_nifty50_symbols() -> list[str] | None:
    """
    Fetch the latest NIFTY 50 constituents from NSE's official CSV.
    Returns a list of SYMBOL strings (e.g. ['RELIANCE', 'HDFCBANK', ...])
    or None on failure.

    Uses a browser-like User-Agent and Referer to avoid NSE blocking the request.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/",
        "Accept": "text/csv,application/vnd.ms-excel,application/octet-stream;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(NIFTY50_CSV_URL, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.warning(f"Failed to download NIFTY 50 CSV from NSE: {e}")
        return None

    try:
        # CSV usually has columns like: Company Name, Industry, Symbol, Series, ...
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        st.warning(f"Failed to parse NIFTY 50 CSV: {e}")
        return None

    if "Symbol" not in df.columns:
        st.warning("NSE NIFTY 50 CSV does not contain 'Symbol' column. Format may have changed.")
        return None

    symbols = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )

    # Sort just to have stable ordering
    symbols = sorted(symbols)

    if len(symbols) != 50:
        st.warning(f"Expected 50 NIFTY symbols, got {len(symbols)}. "
                   "Proceeding, but please verify against NSE manually.")

    return symbols
