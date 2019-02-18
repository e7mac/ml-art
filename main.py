from audio_tools import *
from video_tools import *
from file_tools import *
import os
from requests import get
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mp4"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mp4"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mp4"
# download(sample, 'input/raw')

filename = "input/raw/BreakMeMadeira-small.mp4"

def preprocess(filename):
    makeMainDirectories()

    extractAudio(filename)
    audioSampleRate, audioSignal = scipy.io.wavfile.read(filename + ".wav")
    print(audioSignal)

    videoArray = videoRead(filename)
    print(videoArray)

def load(filename):
    filename = "input/raw/BreakMeMadeira-small.mp4"
    audioSampleRate, audioSignal = scipy.io.wavfile.read(filename + ".wav")
    videoArray = videoRead(filename)
    return videoArray, audioSignal

def linear(x, n_output, name=None, activation=None, reuse=None):
    """Fully connected layer

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    op : tf.Tensor
        Output of fully connected layer.
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W

def process(filename):
    v = VideoDataSet(filename)
    a = AudioDataSet(v.audioFilename())
    return a, v

def createNetwork():
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 480, 720, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 2])
    n_neurons = 40

    h = X
    for i in range(20):
        h, W = linear(x=h, n_output=n_neurons, name='layer'+str(i), activation=tf.nn.relu)

    # Now, make one last layer to make sure your network has 3 outputs:
    Y_pred, W_final = linear(h, 3, activation=None, name='pred')


# preprocess()
# load()
a, v = process(filename)



