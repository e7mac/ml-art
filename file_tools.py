from audio_tools import *
from video_tools import *
import os
from requests import get
from pathlib import Path
import moviepy.editor as mp
import scipy.io.wavfile

def createDirectory(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)

def makeMainDirectories():
    createDirectory('input')
    createDirectory('input/raw')
    createDirectory('input/preprocessed')
    createDirectory('output')

def download(url, dirName):
    print('here')
    file_name = dirName + '/' + url.split('/')[-1]
    print(file_name)
    my_file = Path(file_name)
    if not my_file.is_file():
        with open(file_name, "wb") as file:
            print('Downloading file....')
            response = get(url)
            file.write(response.content)
