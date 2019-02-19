import tensorflow as tf


def _encoder_block(inputs, filters):
  """
  """
  outputs = tf.keras.layers.Conv2D(
    filters, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)(inputs)
  outputs = tf.keras.layers.Conv2D(
    filters, 3, strides=2, padding='same', activation=tf.nn.leaky_relu)(outputs)
  return outputs


def image_encoder(input_images, output_dims):
  """
  """
  outputs = _encoder_block(input_images, 32)
  outputs = _encoder_block(outputs, 64)
  outputs = _encoder_block(outputs, 128)
  outputs = _encoder_block(outputs, 256)
  outputs = _encoder_block(outputs, 512)
  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Dense(output_dims, activation=None)(outputs)
  return outputs