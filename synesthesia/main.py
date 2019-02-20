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


def _infer(output, feed_dict):
    """
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(output, feed_dict=feed_dict)

def _train(train_op, feed_dict, train_dir, max_steps=1000, summary_steps=10, 
           log_steps=10, save_checkpoint_secs=180):
    """
    """
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    global_step = tf.train.get_or_create_global_step()
    logging_tensors = {
        'step': global_step, 'loss': tf.losses.get_total_loss()}
    hooks = [tf.train.StopAtStepHook(max_steps)]
    logging_hook = tf.train.LoggingTensorHook(
        logging_tensors, every_n_iter=log_steps)
    hooks += [logging_hook]
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,
        hooks=hooks,
        save_checkpoint_secs=save_checkpoint_secs,
        save_summaries_steps=summary_steps,
        log_step_count_steps=log_steps,
        config=session_config) as sess:
      while not sess.should_stop():
        sess.run(train_op, feed_dict=feed_dict)

def _get_data():
    """
    """
    makeMainDirectories()

    sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mp4"
    # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mp4"
    # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mp4"
    # sample = "https://www.sample-videos.com/video123/mp4/240/big_buck_bunny_240p_20mb.mp4"
    download(sample, 'input/raw')
    filename = "input/raw/BreakMeMadeira-small.mp4"
    # filename = "input/raw/big_buck_bunny_240p_20mb.mp4"

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
    # vs = np.expand_dims(np.concatenate(vs, axis=-1), 0)
    # audios = np.expand_dims(np.concatenate(audios, axis=0), 0)    
    return vs, audios

def writeWaveFile(outputfile, sampleRate, data):
    import scipy.io.wavfile
    scipy.io.wavfile.write(outputfile, sampleRate, data)

def main():
    args = sys.argv
    output_directory = "."

    for arg in args:
        split = arg.split("--output=")
        if len(split) > 1:
            output = split[-1]
            output_directory = output

    # Set up data
    batchSize = 8
    inputs = tf.placeholder(
        tf.float32, [batchSize, _IMAGE_CROP_SIZE, _IMAGE_CROP_SIZE, 3 * _NUM_TEMPORAL_FRAMES])
    targets = tf.placeholder(
        tf.float32, [batchSize, _AUDIO_DIMS * _NUM_TEMPORAL_FRAMES])
    av = _get_data()
    # Create model.
    outputs = networks.image_encoder(
        inputs, _AUDIO_DIMS * _NUM_TEMPORAL_FRAMES)
    
    # Add losses.
    l1_loss = tf.losses.absolute_difference(targets, outputs)
    tf.losses.add_loss(l1_loss)
    tf.summary.scalar('l1_loss', l1_loss)

    # Train!
    optimizer = tf.train.AdamOptimizer(_LEARNING_RATE)
    loss_op = tf.losses.get_total_loss()
    train_op = tf.contrib.training.create_train_op(loss_op, optimizer)

    # Batch 
    for batch in range(int((av.video.frameCount - 1 - _NUM_TEMPORAL_FRAMES) / batchSize)):
        images_arr, audios_arr = getTemporalFramesBatchFromData(av, batch, batchSize)
        feed_dict = {
            inputs: images_arr,
            targets: audios_arr
        }
        _train(
            train_op, 
            feed_dict,
            train_dir=output_directory + '/tmp/image_to_sound/train')
    print('training done')
    audio_pred_array = []
    # Infer 
    for batch in range(int((av.video.frameCount - 1 - _NUM_TEMPORAL_FRAMES) / batchSize)):
        images_arr, audios_arr = getTemporalFramesBatchFromData(av, batch, batchSize)
        feed_dict = {
            inputs: images_arr,
            targets: audios_arr
        }
        inference = _infer(outputs, feed_dict)
        print(inference.shape)
        print(inference.dtype)
        print(inference)
        print('infer done')
        audio_pred_array.append(inference.tolist())
        print('append done')
    audio_pred_array = np.array(audio_pred_array).flatten()
    print('np flatten done')
    print(audio_pred_array)
    print(audio_pred_array.shape)
    print(audio_pred_array.dtype)
    writeWaveFile(output_directory + '/output.wav', av.audio.sampleRate, audio_pred_array)
    print('write wave done')
    
if __name__ == "__main__":
    main()
