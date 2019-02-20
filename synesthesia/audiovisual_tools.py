from pydub import AudioSegment
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
from audio_tools import *
from video_tools import *

class AudiovisualDataSet():
    """A class to store audiovisual data set for AI processing"""
    audio = None
    video = None

    def __init__(self, filename):
        self.video = VideoDataSet(filename)
        self.video.extractAudio()
        self.audio = AudioDataSet(self.video.audioFilename())

    def printStats(self):
        print(self.video.frameRate)
        print(self.video.frameCount)
        print(self.video.frameCount / self.video.frameRate)
        
        print(self.audio.sampleRate)
        print(len(self.audio.signal))
        print(len(self.audio.signal) / self.audio.sampleRate)

    def slice(self, frameIndex):
        if frameIndex > self.video.frameCount:
            # add error
            assert(frameIndex < self.video.frameCount)    
        return self.video.imgs[frameIndex], self.audioForFrame(frameIndex)

    def audioForFrame(self, frameIndex):
        duration = 1. / self.video.frameRate
        numAudioFrames =  int(duration * self.audio.sampleRate)
        startAudioSampleIndex = frameIndex * numAudioFrames
        return self.audio.signal[startAudioSampleIndex:startAudioSampleIndex+numAudioFrames]