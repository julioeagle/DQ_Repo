import pickle
import os
from pprint import pprint as pp
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pandas as pd

def Create_Service(client_secret_file, api_name, api_version, scopes):
    print(client_secret_file,api_name, api_version, scopes, sep='-')

    
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]
    #SCOPES = scopes[0]
    print("EL SCOPE ES", SCOPES)
    
    cred = None
    
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            cred = pickle.load(token)
    
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()
    
        with open('token.pickle', 'wb') as token:
            pickle.dump(cred, token)
    
    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME,'Service created successfully')
        return service
    except Exception as e:
        print(e)
        return None

