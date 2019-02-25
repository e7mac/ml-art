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
_AUDIO_DIMS = 1./30 * 44100 #per frame for 30 fps video
_IMAGE_CROP_SIZE = 200
_LEARNING_RATE = 1e-6


def _infer(output, feed_dict):
    """
    """
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        return sess.run(output, feed_dict=feed_dict)

def _get_model_init_fn(checkpoint_load_path):
 """Constructs and returns a restore_fn from a given checkpoint path."""
 variables_to_restore = tf.contrib.framework.get_variables_to_restore(
     exclude=['global_step'])
 if variables_to_restore:
   init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
       checkpoint_load_path, variables_to_restore, ignore_missing_vars=True)
   global_step = tf.train.get_or_create_global_step()

   def restore_fn(unused_scaffold, sess):
     sess.run(init_op, init_feed_dict)
     sess.run([global_step])

   return restore_fn
 return None

def _train(train_op, feed_dict, train_dir, max_steps=500, summary_steps=10, 
           log_steps=10, save_checkpoint_secs=180, checkpoint_load_path=None):
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
    init_fn = None
    if checkpoint_load_path:
        init_fn = _get_model_init_fn(checkpoint_load_path)
    summary_op = tf.summary.merge_all()
    scaffold = tf.train.Scaffold(
        init_fn=init_fn,
        summary_op=summary_op,
    )
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,
        hooks=hooks,
        scaffold=scaffold,
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

    # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mp4"
    sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mp4"
    # sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mp4"
    download(sample, 'input/raw')
    # filename = "input/raw/BreakMeMadeira-small.mp4"
    filename = "input/raw/BreakMeMadeira-medium.mp4"
    # filename = "input/raw/BreakMeMadeira-full.mp4"

    av = AudiovisualDataSet(filename)
    return av
def getTemporalFramesBatchFromData(av, startFrame, batchSize):
    vs = []
    audios = []
    for index in range(batchSize):
        v, a = getTemporalFramesFromData(av, startFrame + index)
        vs.append(v)
        audios.append(a)
    vs = np.array(vs)
    audios = np.array(audios)
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
    audioArray = np.array(audioArray, dtype=np.float32).flatten()
    writeWaveFile(filename, sampleRate, audioArray)

def startTensorboard():
    os.system('tensorboard --logdir=/Users/mayank/ml-art')

def stopTensorboard(process):
    process.terminate()
    os.system('kill ' + str(process.pid))

def main():
    args = sys.argv
    output_directory = "."
    import datetime
    run = datetime.datetime.now().strftime("%m-%d-%Y %I-%M%p")

    for arg in args:
        split = arg.split("--output=")
        if len(split) > 1:
            output_directory = split[-1]

        split = arg.split("--run=")
        if len(split) > 1:
            run = split[-1]

    # Set up data
    batchSize = 8
    inputs = tf.placeholder(
        tf.float32, [None, _IMAGE_CROP_SIZE, _IMAGE_CROP_SIZE, 3 * _NUM_TEMPORAL_FRAMES], name='inputs')
    targets = tf.placeholder(
        tf.float32, [None, _AUDIO_DIMS * _NUM_TEMPORAL_FRAMES], name='targets')
    av = _get_data()
    # Create model.
    outputs = networks.image_encoder(
        inputs, _AUDIO_DIMS * _NUM_TEMPORAL_FRAMES)
    tf.summary.audio('outputs', outputs, av.audio.sampleRate)
    tf.summary.audio('targets', targets, av.audio.sampleRate)
    
    # Add losses.
    l1_loss = tf.losses.absolute_difference(targets, outputs)
    l2_loss = tf.losses.mean_squared_error(targets, outputs)
    tf.losses.add_loss(l2_loss)
    tf.summary.scalar('l2_loss', l2_loss)

    # Train!

    from multiprocessing import Process
    createDirectory(output_directory + '/train/' + run)
    process = Process(target = startTensorboard)
    process.start()

    optimizer = tf.train.AdamOptimizer(_LEARNING_RATE)
    loss_op = tf.losses.get_total_loss()
    train_op = tf.contrib.training.create_train_op(loss_op, optimizer)

    from random import shuffle
    # Batch 
    batches = list(range(int((av.video.frameCount - _NUM_TEMPORAL_FRAMES) / batchSize)))
    shuffle(batches)
    for batch in batches:
        images_arr, audios_arr = getTemporalFramesBatchFromData(av, batch*batchSize, batchSize)
        feed_dict = {
            inputs: images_arr,
            targets: audios_arr
        }
        _train(
            train_op, 
            feed_dict,
            train_dir=output_directory + '/train/' + run)
    print('training done')
    stopTensorboard(process)
    
if __name__ == "__main__":
    main()
