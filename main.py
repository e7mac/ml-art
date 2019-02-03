from audio_tools import *
from video_tools import *
from file_tools import *
import os
from requests import get
from pathlib import Path

makeMainDirectories()

sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mp4"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mp4"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mp4"
download(sample, 'input/raw')


filename = "input/raw/BreakMeMadeira-small.mp4"
extractAudio(filename)
audioSampleRate, audioSignal = scipy.io.wavfile.read(filename + ".wav")
print(audioSignal)

videoArray = videoRead(filename)
print(videoArray)
