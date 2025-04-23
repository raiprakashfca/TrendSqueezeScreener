import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def load_credentials_from_gsheet(sheet_name):
    # Define scopes
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Use gcp_service_account directly from st.secrets
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)

    # Read API credentials from Google Sheet
    sheet = client.open(sheet_name).sheet1
    api_key = sheet.cell(1, 1).value
    api_secret = sheet.cell(1, 2).value
    access_token = sheet.cell(1, 3).value

    return api_key, api_secret, access_token
