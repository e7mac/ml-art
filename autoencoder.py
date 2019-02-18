from audio_tools import *
from video_tools import *
from file_tools import *
import os
from requests import get
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import utils

class Autoencoder():
    """Class that creates an autoencoder"""

    def __init__(self, v, a, output_directory):
        self.output_directory = output_directory
        self.createNetwork(v, a)
        self.train(v, a)

    def train(self, v, a):
        # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # Some parameters for training
        batch_size = 10
        n_epochs = 60
        step = 10

        # We'll try to reconstruct the same first 10 images and show how
        # The network does over the course of training.
        examples_v = v.imgs[:10]
        # examples_a = a.signal[:10]

        # We have to preprocess the images before feeding them to the network.
        # I'll do this once here, so we don't have to do it every iteration.
        test_v = v.preprocess(examples_v).reshape(-1, v.n_features())
        # test_a = a.preprocess(examples_a).reshape(-1, a.n_features)

        # If we want to just visualize them, we can create a montage.
        test_images = utils.montage(examples_v, saveto=self.output_directory+'/input_montage.png')#.astype(np.uint8)

        # Store images so we can make a gif
        gifs = []
        # Now for our training:
        for epoch_i in range(n_epochs):
            # Keep track of the cost
            this_cost = 0


            # # Iterate over the entire dataset in batches
            # for batch_X, _ in ds.train.next_batch(batch_size=batch_size):
                
            #     # (TODO) Preprocess and reshape our current batch, batch_X:
            #     this_batch = preprocess(batch_X, ds).reshape(-1, n_features)
                
            #     # Compute the cost, and run the optimizer.
            #     this_cost += sess.run([cost, optimizer], feed_dict={X: this_batch})[0]
            
            # Average cost of this epoch
            # avg_cost = this_cost / ds.X.shape[0] / batch_size
            # X1 = tf.get_default_graph().get_tensor_by_name('X1')
            # X2 = tf.get_default_graph().get_tensor_by_name('X2')
            X1 = self.X1
            X2 = self.X2
            avg_cost = sess.run([self.cost, self.optimizer], feed_dict={X1: v.imgs.reshape(-1, v.n_features()), X2: a.signal.reshape(-1, a.n_features())})
            print(epoch_i, avg_cost)
            
            # Let's also try to see how the network currently reconstructs the input.
            # We'll draw the reconstruction every `step` iterations.
            if epoch_i % step == 0:
                
                Y1 = self.Y1
                Y2 = self.Y2
                # (TODO) Ask for the output of the network, Y, and give it our test examples

                recon = sess.run(Y1, feed_dict={X1: v.imgs.reshape(-1, v.n_features()), X2: a.signal.reshape(-1, a.n_features())})
                                
                # Resize the 2d to the 4d representation:
                rsz = recon.reshape(v.imgs.shape)

                # We have to unprocess the image now, removing the normalization
                unnorm_img = v.deprocess(rsz)
                                
                # Clip to avoid saturation
                # TODO: Make sure this image is the correct range, e.g.
                # for float32 0-1, you should clip between 0 and 1
                # for uint8 0-255, you should clip between 0 and 255!
                clipped = np.clip(unnorm_img, 0, 255)

                # And we can create a montage of the reconstruction
                recon = utils.montage(clipped, saveto=self.output_directory+str(epoch_i)+".png")
                
                # Store for gif
                gifs.append(recon)

                fig, axs = plt.subplots(1, 2, figsize=(10, 10))
                axs[0].imshow(test_images)
                axs[0].set_title('Original')
                axs[1].imshow(recon)
                axs[1].set_title('Synthesis')
                fig.canvas.draw()
        plt.show()
        

    def createNetwork(self, v, a):
        tf.reset_default_graph()
        X1 = tf.placeholder(np.float32, name='X1', shape=(None, v.n_features()))
        X2 = tf.placeholder(np.float32, name='X2', shape=(None, a.n_features()))
        self.X1 = X1
        self.X2 = X2
        encoder_dimensions = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
        self.W1s, self.W2s, self.z = self.encode(X1, X2, encoder_dimensions)
        # [print(op.name) for op in tf.get_default_graph().get_operations()]
        # print(v.n_features())
        # print(a.n_features())
        # [print(W1_i.get_shape().as_list()) for W1_i in self.W1s]
        # [print(W2_i.get_shape().as_list()) for W2_i in self.W2s]
        # print(self.z.get_shape().as_list())

        # We'll first reverse the order of our weight matrices
        self.decoder_W1s = self.W1s[::-1]
        self.decoder_W2s = self.W2s[::-1]
        # then reverse the order of our dimensions
        # appending the last layers number of inputs.
        decoder1_dimensions = encoder_dimensions[::-1][1:] + [v.n_features()]
        decoder2_dimensions = encoder_dimensions[::-1][1:] + [a.n_features()]
        # print(decoder1_dimensions)
        # print(decoder2_dimensions)

        Y1, Y2 = self.decode(self.z, decoder1_dimensions, decoder2_dimensions, self.decoder_W1s, self.decoder_W2s)

        # [print(op.name) for op in tf.get_default_graph().get_operations() if op.name.startswith('decoder')]
        # print(Y1.get_shape().as_list())
        # print(Y2.get_shape().as_list())

        loss1 = tf.squared_difference(X1, Y1)
        loss2 = tf.squared_difference(X2, Y2)
        self.cost = tf.reduce_mean(tf.reduce_sum(loss1, axis=1)) + tf.reduce_mean(tf.reduce_sum(loss2, axis=1))
        self.Y1 = Y1
        self.Y2 = Y2
        learning_rate = 0.0001
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


    def decode(self, z, dimensions1, dimensions2, W1s, W2s, activation=tf.nn.tanh):
        current_input = z
        
        for layer_i, n_output in enumerate(dimensions1):
            # we'll use a variable scope again to help encapsulate our variables
            # This will simply prefix all the variables made in this scope
            # with the name we give it.
            with tf.variable_scope("decoder/layer/{}".format(layer_i)):

                # Now we'll grab the weight matrix we created before and transpose it
                # So a 3072 x 784 matrix would become 784 x 3072
                # or a 256 x 64 matrix, would become 64 x 256
                W = tf.transpose(W1s[layer_i])

                # Now we'll multiply our input by our transposed W matrix
                h = tf.matmul(current_input, W)

                # And then use a relu activation function on its output
                current_input = activation(h)

                # We'll also replace n_input with the current n_output, so that on the
                # next iteration, our new number inputs will be correct.
                n_input = n_output                
        Y1 = current_input

        current_input = z
        for layer_i, n_output in enumerate(dimensions2):
            # we'll use a variable scope again to help encapsulate our variables
            # This will simply prefix all the variables made in this scope
            # with the name we give it.
            with tf.variable_scope("decoder/layer/{}".format(layer_i)):

                # Now we'll grab the weight matrix we created before and transpose it
                # So a 3072 x 784 matrix would become 784 x 3072
                # or a 256 x 64 matrix, would become 64 x 256
                W = tf.transpose(W2s[layer_i])

                # Now we'll multiply our input by our transposed W matrix
                h = tf.matmul(current_input, W)

                # And then use a relu activation function on its output
                current_input = activation(h)

                # We'll also replace n_input with the current n_output, so that on the
                # next iteration, our new number inputs will be correct.
                n_input = n_output                
        Y2 = current_input
        return Y1, Y2


    def linear(self, x, n_output, name=None, activation=None, reuse=None):
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


    def encode(self, X1, X2, dimensions, activation=tf.nn.tanh):
        W1s = []
        W2s = []
        # We'll create a for loop to create each layer:
        for layer_i, n_output in enumerate(dimensions):
            with tf.variable_scope('encoder'):
                h1, W1 = self.linear(X1, n_output, name='1layer' + str(layer_i), activation=activation)
                h2, W2 = self.linear(X2, n_output, name='2layer' + str(layer_i), activation=activation)
                W1s.append(W1)
                W2s.append(W2)
                X1 = h1
                X2 = h2
        
        z = X1 + X2
        return W1s, W2s, z
