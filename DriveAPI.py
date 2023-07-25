from __future__ import print_function
import pickle
import os.path
import io
import shutil
from mimetypes import MimeTypes
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload


class DriveAPI:

    def __init__(self, net_type, board_size):
        self.net_type = net_type
        self.board_size = board_size
        # define the SCOPES
        self.SCOPES = ['https://www.googleapis.com/auth/drive']

        # Variable self.creds will
        # store the user access token.
        # If no valid token found
        # we will create one.
        self.creds = None

        # The file token.pickle stores the
        # user's access and refresh tokens. It is
        # created automatically when the authorization
        # flow completes for the first time.

        # Check if file token.pickle exists
        if os.path.exists('token.pickle'):
            # Read the token from the file and
            # store it in the variable self.creds
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token)

        # If no valid credentials are available,
        # request the user to log in.
        if not self.creds or not self.creds.valid:

            # If token is expired, it will be refreshed,
            # else, we will request a new one.
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                self.creds = flow.run_local_server(port=0)

            # Save the access token in token.pickle
            # file for future usage
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)

        self.update_file_list()

    def FileDownload(self, file_id, file_name):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()

        prev_dir = os.getcwd()
        os.chdir(f'logs/go/{self.net_type}_MCTS_SimModified_checkpoint/{self.board_size}/')

        # Initialise a downloader object to download the file
        downloader = MediaIoBaseDownload(fh, request, chunksize=204800)
        done = False

        try:
            # Download the data in chunks
            while not done:
                status, done = downloader.next_chunk()

            fh.seek(0)
            # Write the received data to the file
            with open(file_name, 'wb') as f:
                shutil.copyfileobj(fh, f)

            os.chdir(prev_dir)
            # Return True if file Downloaded successfully
            self.service.files().delete(fileId=file_id).execute()
            return True
        except:

            # Return False if something went wrong
            print("Something went wrong with download.")
            return False

    def FileUpload(self, filepath, modelnum):

        # Extract the file name out of the file path
        # name = filepath.split('/')[-1]
        # print(name)
        name = f'best{modelnum}.pth.tar'

        # Find the MimeType of the file
        mimetype = MimeTypes().guess_type(name)[0]

        # create file metadata
        file_metadata = {'name': name}

        try:
            media = MediaFileUpload(filepath, mimetype=mimetype, resumable=True)

            # Create a new file in the Drive storage
            file = self.service.files().create(
                body=file_metadata, media_body=media, fields='id').execute()

            print("File Uploaded.")

        except:
            pass

    def update_file_list(self):
        # Connect to the API service
        self.service = build('drive', 'v3', credentials=self.creds)

        # request a list of first N files or
        # folders with name and id from the API.
        results = self.service.files().list(
            pageSize=1000, fields="files(id, name)").execute()
        self.items = results.get('files', [])

        # print a list of files
        # print("First Item:",  items[1][0].id,"\n")
        # print(*self.items, sep="\n", end="\n\n")
        # print(f"{len(self.items)} Files Found on Drive")
        # print("done printing items")
