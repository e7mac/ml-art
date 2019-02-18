from audio_tools import *
from video_tools import *
from file_tools import *
from autoencoder import *

import os
import sys

import tensorflow as tf

def process(filename):
    v = VideoDataSet(filename)
    v.extractAudio()
    a = AudioDataSet(v.audioFilename())
    return v, a

# preprocess()
# load()


def main():
    args = sys.argv
    output_directory = "."

    for arg in args:
        split = arg.split("--output=")
        if len(split) > 1:
            output = split[-1]
            output_directory = output
    
    makeMainDirectories()

    sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mp4"
    # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mp4"
    # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mp4"
    download(sample, 'input/raw')

    filename = "input/raw/BreakMeMadeira-small.mp4"

    v, a = process(filename)
    a = Autoencoder(v, a, output_directory)

if __name__ == "__main__":
    main()