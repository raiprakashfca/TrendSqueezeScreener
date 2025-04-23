import gspread
from oauth2client.service_account import ServiceAccountCredentials

def load_credentials_from_gsheet(sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("gcreds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).sheet1

    api_key = sheet.cell(1, 1).value
    api_secret = sheet.cell(1, 2).value
    access_token = sheet.cell(1, 3).value
    return api_key, api_secret, access_token
