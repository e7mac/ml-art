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
    file_name = dirName + '/' + url.split('/')[-1]
    print(file_name)
    my_file = Path(file_name)
    if my_file.is_file():
        with open(file_name, "wb") as file:
            print('Downloading file....')
            response = get(url)
            file.write(response.content)

def extractAudio(filename):
    clip = mp.VideoFileClip(filename)
    clip.audio.write_audiofile(filename + ".wav")

makeMainDirectories()

# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mov"
# # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mov"
# # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mov"
# download(sample, 'input/raw')

# timeSlice = 1 #seconds
filename = "input/raw/BreakMeMadeira-small.mp4"
# extractAudio(filename)

audioSampleRate, audioSignal = scipy.io.wavfile.read(filename + ".wav")
print(audioSignal)

videoArray = videoRead(filename)
print(videoArray)

# audioSlice("input/raw/BreakMeMadeira-small.mp4", timeSlice)
# videoSlice("input/raw/BreakMeMadeira-small.mp4", timeSlice, 10, 0.1)
