from audio_tools import *
from video_tools import *
from audiovisual_tools import *
from file_tools import *
from autoencoder import *
import networks

import os
import sys

import tensorflow as tf

_NUM_TEMPORAL_FRAMES = 5
_AUDIO_DIMS = 1470
_IMAGE_CROP_SIZE = 256
_LEARNING_RATE = 1e-5

sess = tf.Session()
saver = tf.train.import_meta_graph('/storage/tmp/image_to_sound/train/model.ckpt-53250.meta')
saver.restore(sess,tf.train.latest_checkpoint('/storage/learning1e6/tmp/image_to_sound/train/'))
inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
targets = tf.get_default_graph().get_tensor_by_name('targets:0')
outputs = tf.get_default_graph().get_tensor_by_name('outputs/BiasAdd:0')

def _infer(i, t):
    """
    """
    return sess.run(o, feed_dict={
        inputs: i,
        targets: t
    })

def _get_data():
    """
    """
    makeMainDirectories()

#     sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mp4"
    sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mp4"
    # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mp4"
    download(sample, 'input/raw')
    filename = "input/raw/BreakMeMadeira-medium.mp4"

    av = AudiovisualDataSet(filename)
    return av

def getTemporalFramesBatchFromData(av, frame, batchSize):
    vs = []
    audios = []
    for index in range(batchSize):
        v, a = getTemporalFramesFromData(av, frame + index)
        vs.append(v)
        audios.append(a)
    vs = np.array(vs)
    audios = np.array(audios)
    print(vs.shape)
    print(audios.shape)
    return vs, audios

def getTemporalFramesFromData(av, startFrame):
    vs = []
    audios = []
    for i in range(_NUM_TEMPORAL_FRAMES):
        v, a = av.slice(startFrame + i)
        vs.append(cv2.resize(v, (_IMAGE_CROP_SIZE, _IMAGE_CROP_SIZE)))
        audios.append(a[:,0])
    vs = np.concatenate(vs, axis=-1)
    audios = np.concatenate(audios, axis=0)
    return vs, audios
def writeWaveFile(outputfile, sampleRate, data):
    import scipy.io.wavfile
    scipy.io.wavfile.write(outputfile, sampleRate, data)

def flattenAudioAndWriteWav(audioArray, filename, sampleRate):
    print("---")
    print(audioArray)
    print(audioArray.shape)
    print(audioArray.dtype)
    audioArray = np.array(audioArray, dtype=np.float32).flatten()
    print(audioArray)
    print(audioArray.shape)
    print(audioArray.dtype)
    writeWaveFile(filename, sampleRate, audioArray)

av = _get_data()

images_arr, audios_arr = getTemporalFramesBatchFromData(av, 10, 8)
plt.imshow(av.video.imgs[0])

batchSize = 8
for batch in range(int((av.video.frameCount - 1 - _NUM_TEMPORAL_FRAMES) / batchSize)):
    images_arr, audios_arr = getTemporalFramesBatchFromData(av, batch, batchSize)
    inference = _infer(None, images_arr, audios_arr)
    flattenAudioAndWriteWav(inference, 'output/learning1e6-inference'+str(batch)+'.wav', av.audio.sampleRate)