import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

def load_credentials_from_gsheet(sheet_name):
    # Load Google service account JSON from Streamlit secrets
    gcp_json = json.loads(st.secrets["gcp_service_account"])

    # Define the scopes required for accessing Google Sheets
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Create credentials from dict and authorize gspread
    creds = ServiceAccountCredentials.from_json_keyfile_dict(gcp_json, scope)
    client = gspread.authorize(creds)

    # Open the first worksheet in the specified sheet
    sheet = client.open(sheet_name).sheet1

    # Read API Key, Secret, and Access Token from top row
    api_key = sheet.cell(1, 1).value
    api_secret = sheet.cell(1, 2).value
    access_token = sheet.cell(1, 3).value

    return api_key, api_secret, access_token
