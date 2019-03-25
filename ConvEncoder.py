import os, sys
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import math

import matplotlib
import matplotlib.pyplot as plt

path = "./data/BBBC016_v1_images"
#input_dim = 262144
input_dim = 20000
alpha = 0.005
num_epochs = 200
batch_size = 16
picture_dim = 128

print("Loading data......")
cell_images = []
for filename in os.listdir(path):
    im = Image.open(path + "/" + filename)
    imarray = np.array(im)[:128,:128]
    cell_images.append(imarray.reshape(picture_dim, picture_dim, 1))
print(np.array(cell_images).shape)
print("Loaded data.")

def get_placeholders():
	inputs_placeholder = tf.placeholder(tf.float32, (None, picture_dim, picture_dim, 1))
	labels_placeholder = tf.placeholder(tf.float32, (None, picture_dim, picture_dim, 1))
	return inputs_placeholder, labels_placeholder

def encoder(inputs_batch):
	conv1 = tf.layers.conv2d(inputs=inputs_batch, filters=12, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
	conv2 = tf.layers.conv2d(inputs=maxpool1, filters=12, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
	conv3 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	maxpool3 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
	return maxpool3

def decoder(inputs_batch):
	deconv1 = tf.layers.conv2d(inputs=inputs_batch, filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	upsample1 = tf.image.resize_images(deconv1, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	deconv2 = tf.layers.conv2d(inputs=upsample1, filters=12, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	upsample2 = tf.image.resize_images(deconv2, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	deconv3 = tf.layers.conv2d(inputs=upsample2, filters=12, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	upsample3 = tf.image.resize_images(deconv3, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	output = tf.layers.conv2d(inputs=upsample3, filters=1, kernel_size=(3,3), padding='same', activation=None)
	return output

def compute_loss(y_hat, labels_batch):
    for img in y_hat:
        

    loss = tf.reduce_mean(tf.pow(y_hat - labels_batch, 2))
    return loss

def get_batches(seq, size=batch_size):
	return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def plot_loss(cost_list):
    plt.title("Generator Loss Curve")
    plt.plot(cost_list, '-')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def train(X, Y):
	inputs_batch, labels_batch = get_placeholders()
	encoding = encoder(inputs_batch)
	y_hat = decoder(encoding)
	loss = compute_loss(y_hat, labels_batch)
	optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(loss)
	init = tf.global_variables_initializer()

    global curr_iter
	with tf.Session() as sess:
		sess.run(init)
        cost_list = []
		for iteration in range(num_epochs):
            curr_iter += 1
			inputs_batches = get_batches(X)
			for i in range(len(inputs_batches)):
				batch = inputs_batches[i]
				bottleneck, preds, _, curr_loss = sess.run([encoding, y_hat, optimizer, loss], feed_dict={inputs_batch: batch, labels_batch: batch})
				_, _, total_loss = sess.run([encoding, y_hat, loss], feed_dict={inputs_batch : X, labels_batch : X})
				print ("Epoch " + str(iteration+1) + ", Update Number " + str(i)+ ", Curr Cost : "  + str(total_loss))
                cost_list.append(total_loss)

    plot_loss(cost_list)

train(cell_images, cell_images)
