from __future__ import print_function
import pickle
import os.path
import io
import shutil
import requests
from mimetypes import MimeTypes
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import json

BoardSize = 7
NetType = 'RES' #or 'CNN'
tag = 'MCTS_SimModified'

args = dict({
	'numIters': 1000,
	'numEps': 20,              # Number of complete self-play games to simulate during a new iteration.
	'tempThreshold': 15,
	'updateThreshold': 0.54,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
	'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
	'numMCTSSims': 150,         # Number of games moves for MCTS to simulate.
	'arenaCompare': 3,         # Number of games to play during arena play to determine if new net will be accepted.
	'cpuct': 1.0,

	# 10 process maximum currently, do not set any higher
	'num_processes': 4,

	'checkpoint': './logs/go/{}_checkpoint/{}/'.format(NetType + '_' + tag, BoardSize),
	'load_model': False,
	'numItersForTrainExamplesHistory': 25,
})

class DriveAPI:
	global SCOPES
	
	# Define the scopes
	SCOPES = ['https://www.googleapis.com/auth/drive']

	def __init__(self):
		
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
					'credentials.json', SCOPES)
				self.creds = flow.run_local_server(port=0)

			# Save the access token in token.pickle
			# file for future usage
			with open('token.pickle', 'wb') as token:
				pickle.dump(self.creds, token)

		# Connect to the API service
		self.service = build('drive', 'v3', credentials=self.creds)

		# request a list of first N files or
		# folders with name and id from the API.
		results = self.service.files().list(
			pageSize=1000, fields="files(id, name)").execute()
		self.items = results.get('files', [])
		
		# print a list of files
		#print("First Item:",  items[1][0].id,"\n")
		print(*self.items, sep="\n", end="\n\n")
		print("done printing items")
		"""for item in items:
			if item['name'] == 'Sabaki_Engine':
				checkpoint = item
				break
			else:
				checkpoint = items[0]
		print(checkpoint)"""

	def FileDownload(self, file_id, file_name):
		
		

		request = self.service.files().get_media(fileId=file_id)
		fh = io.BytesIO()
		
		prev_dir = os.getcwd()
		print(os.getcwd())
		os.chdir(f'logs/go/{NetType}_MCTS_SimModified_checkpoint/{BoardSize}/')
		print(os.getcwd())

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
			return True
		except:
			
			# Return False if something went wrong
			print("Something went wrong with download.")
			return False

	def FileUpload(self, filepath, modelnum):
		
		# Extract the file name out of the file path
		#name = filepath.split('/')[-1]
		#print(name)
		name = f'best{modelnum}.pth.tar'
		"""if len(name) == 1:
			name = filepath.split('\\')[-1]"""
		
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
			# Raise UploadError if file is not uploaded.
			#raise UploadError("Can't Upload File.")