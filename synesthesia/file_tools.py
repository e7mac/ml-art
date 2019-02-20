import os
from pathlib import Path
import requests

def createDirectory(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)

def makeMainDirectories():
    createDirectory('input')
    createDirectory('input/raw')
    createDirectory('input/preprocessed')
    createDirectory('output')

def download(url, dirName):
    file_name = dirName + '/' + url.split('/')[-1]
    my_file = Path(file_name)
    if not my_file.is_file():
        with open(file_name, "wb") as file:
            print('Downloading file....')
            response = requests.get(url)
            file.write(response.content)
