# from audio_tools import *
# from video_tools import *
import os
from requests import get  # to make GET request

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
    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)

makeMainDirectories()

sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mov"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mov"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mov"
download(sample, 'input/raw/')

timeSlice = 1 #seconds

# audioSlice("BreakMeMadeira31.mov", timeSlice)
# videoSlice("BreakMeMadeira31.mov", timeSlice, 10, 0.1)
