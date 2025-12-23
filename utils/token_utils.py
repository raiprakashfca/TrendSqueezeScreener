# utils/token_utils.py
"""
Zerodha credential loader using Google Sheets, with gcp_service_account
stored as a JSON string in Streamlit secrets.
"""

import json
from typing import Tuple

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st


def _get_gspread_client_from_secrets() -> gspread.Client:
    """
    Build a gspread client from st.secrets['gcp_service_account'].

    In your Streamlit Cloud, gcp_service_account is stored as a triple-quoted
    JSON string, so we json.loads() it first.
    """
    raw = st.secrets["gcp_service_account"]
    sa = json.loads(raw) if isinstance(raw, str) else dict(raw)

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa, scope)
    client = gspread.authorize(creds)
    return client


def load_credentials_from_gsheet(sheet_name: str) -> Tuple[str, str, str]:
    """
    Read Zerodha API credentials (api_key, api_secret, access_token)
    from a Google Sheet named `sheet_name`.

    Sheet format (first row as header):
        api_key | api_secret | access_token
        XXXXX   | YYYYY      | ZZZZZ
    """
    client = _get_gspread_client_from_secrets()

    sh = client.open(sheet_name)
    ws = sh.sheet1
    records = ws.get_all_records()

    if not records:
        raise RuntimeError(f"No credentials rows found in sheet '{sheet_name}'")

    row = records[0]
    api_key = row.get("api_key") or row.get("API_KEY")
    api_secret = row.get("api_secret") or row.get("API_SECRET")
    access_token = row.get("access_token") or row.get("ACCESS_TOKEN")

    if not api_key or not access_token:
        raise RuntimeError(
            f"Incomplete credentials in sheet '{sheet_name}'. "
            "Expected columns: api_key, api_secret, access_token"
        )

    return api_key, api_secret, access_token
# utils/token_utils.py
"""
Zerodha credential loader using Google Sheets, with gcp_service_account
stored as a JSON string in Streamlit secrets.
"""

import json
from typing import Tuple

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st


def _get_gspread_client_from_secrets() -> gspread.Client:
    """
    Build a gspread client from st.secrets['gcp_service_account'].

    In your Streamlit Cloud, gcp_service_account is stored as a triple-quoted
    JSON string, so we json.loads() it first.
    """
    raw = st.secrets["gcp_service_account"]
    sa = json.loads(raw) if isinstance(raw, str) else dict(raw)

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa, scope)
    client = gspread.authorize(creds)
    return client


def load_credentials_from_gsheet(sheet_name: str) -> Tuple[str, str, str]:
    """
    Read Zerodha API credentials (api_key, api_secret, access_token)
    from a Google Sheet named `sheet_name`.

    Sheet format (first row as header):
        api_key | api_secret | access_token
        XXXXX   | YYYYY      | ZZZZZ
    """
    client = _get_gspread_client_from_secrets()

    sh = client.open(sheet_name)
    ws = sh.sheet1
    records = ws.get_all_records()

    if not records:
        raise RuntimeError(f"No credentials rows found in sheet '{sheet_name}'")

    row = records[0]
    api_key = row.get("api_key") or row.get("API_KEY")
    api_secret = row.get("api_secret") or row.get("API_SECRET")
    access_token = row.get("access_token") or row.get("ACCESS_TOKEN")

    if not api_key or not access_token:
        raise RuntimeError(
            f"Incomplete credentials in sheet '{sheet_name}'. "
            "Expected columns: api_key, api_secret, access_token"
        )

    return api_key, api_secret, access_token
