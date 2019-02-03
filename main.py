from audio_tools import *
from video_tools import *
import os
from requests import get
from pathlib import Path
import moviepy.editor as mp
import scipy.io.wavfile

def makeMainDirectories():
    createDirectory('input')
    createDirectory('input/raw')
    createDirectory('input/preprocessed')
    createDirectory('output')

makeMainDirectories()

sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mov"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mov"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mov"
download(sample, 'input/raw')


filename = "input/raw/BreakMeMadeira-small.mp4"
extractAudio(filename)
audioSampleRate, audioSignal = scipy.io.wavfile.read(filename + ".wav")
print(audioSignal)

videoArray = videoRead(filename)
print(videoArray)
