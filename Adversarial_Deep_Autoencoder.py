from keras.preprocessing.image import array_to_img
from PIL import Image
import scipy.misc
import utils_conv as utils
import tensorflow.contrib.slim as slim

#Hyperparameters
batch_size   = 30
img_size     = 128
channels     = 3
input_dim    = img_size*img_size*channels
hidden_layer = 1024
latent_dim   = 30
EPOCHS       = 10
ctr = 0

import tensorflow as tf

#Path from where to get the images
dataset = utils.get_image_paths("/home/kunal/fast_style_transfer_per/Coco")

import sys
import numpy as np
import scipy.misc
import os

def lrelu(x, leak = 0.2, name = 'lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1*x + f2*abs(x)

#Defining a convolutional Neural Encoder 
def encoder(image):
    print('Encoder')
    print(image.shape)
    input = slim.conv2d(image, 64, [3,3],
                                stride=[2,2],
                                padding='SAME',
                                weights_initializer = intitializer,
                                activation_fn = lrelu,
                                scope = 'en_conv1')
    print(input.shape)
    net = slim.conv2d(input, 128, [3,3],
                                stride = [2,2],
                                padding = 'SAME',
                                weights_initializer = intitializer,
                                activation_fn = lrelu,
                                scope = 'en_conv2')
    print(net.shape)
    net = slim.conv2d(net, 128, [3,3],
                                stride = [1,1],
                                padding = 'SAME',
                                weights_initializer = intitializer,
                                activation_fn = lrelu,
                                scope = 'en_conv3')
    print(net.shape)
    latent = slim.fully_connected(slim.flatten(net), latent_dim,
                                weights_initializer = intitializer,
                                scope = 'en_latent')
    print(latent.shape)
    return latent
#Defining a convoltuional neural Decoder
def decoder(z, reuse = False):
    print('Decoder')
    print z.shape
    input = slim.fully_connected(z, 32*32*128,
                                weights_initializer = intitializer,
                                activation_fn = lrelu,
                                scope = 'de_flat2')
    print(input.shape)
    net = tf.reshape(input, [-1, 32, 32, 128])
    print(net.shape)
    net = slim.conv2d_transpose(net, 128, [3,3],
                                stride = [1,1],
                                weights_initializer = intitializer,
                                activation_fn = lrelu,
                                scope = 'de_deconv1')
    print(net.shape)
    net = slim.conv2d_transpose(net, 64, [3,3],
                                stride = [2,2],
                                weights_initializer = intitializer,
                                activation_fn = lrelu,
                                scope = 'de_deconv2')
    print(net.shape)
    output = slim.conv2d_transpose(net, 3, [3,3],
                                stride = [2,2],
                                weights_initializer = intitializer,
                                activation_fn = tf.sigmoid,
                                scope = 'de_output')
    print(output.shape)
    return output

#A fully connected discriminator
def discriminator(inputs,reuse = False):
    if reuse:
            tf.get_variable_scope().reuse_variables()
    output = slim.fully_connected(inputs, hidden_layer, 
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'dis_layer1')
    output = slim.fully_connected(output, hidden_layer, 
				weights_initializer = intitializer,
				activation_fn = lrelu,
				scope = 'dis_layer2')
    output = slim.fully_connected(output, 1,
				weights_initializer = intitializer,
				activation_fn = tf.sigmoid,
				scope = 'dis_layer3')
    return output

#Helper function to generate the prior distribution whose shape the latent distribution has to take
def noise(n_samples):
    batch_z = np.random.uniform(-1, 1, [n_samples,latent_dim]).astype(np.float32)
    return batch_z

x = tf.placeholder(tf.float32, [None, img_size, img_size, channels])
intitializer = tf.truncated_normal_initializer(stddev = 0.02)

#Encoder Decoder network
latents = encoder(x)
reconstructions = decoder(latents)

with tf.variable_scope('latent_score') as scope:
    tf.VariableScope.reuse =None
    latent_score = discriminator(latents,reuse = False)

reg_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(latent_score),logits = latent_score))
true_noise_score = discriminator(noise(batch_size),reuse = True)
reconst_cost = tf.reduce_mean(tf.squared_difference(reconstructions,x))

full_enc_cost = 1000*reconst_cost + reg_cost

dec_cost = reconst_cost

discrim_cost  =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(latent_score),logits = latent_score))
discrim_cost +=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(true_noise_score),logits = true_noise_score))

#Defining all the trainable variables
t_vars = tf.trainable_variables()
enc_params  = [var for var in t_vars if 'en_' in var.name]
dec_params  = [var for var in t_vars if 'de_' in var.name]                             
discrim_params = [var for var in t_vars if 'dis_' in var.name]                              
full_cost = full_enc_cost + dec_cost + discrim_cost

lr = tf.placeholder(tf.float32)
e_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(full_enc_cost, var_list=enc_params)
d_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(dec_cost, var_list=dec_params)
dis_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(discrim_cost, var_list=discrim_params)

tf.summary.scalar('Encoder Loss', full_enc_cost)
tf.summary.scalar('Decoder Loss', dec_cost)
tf.summary.scalar('Discriminator Loss', discrim_cost)
tf.summary.scalar('Total Loss', full_cost)

merge = tf.summary.merge_all()
logdir = "./Tensorboard"

session_conf = tf.ConfigProto(gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
                                  allow_soft_placement=True,
                                  log_device_placement=False)

sess = tf.InteractiveSession(config=session_conf)

writer = tf.summary.FileWriter(logdir, sess.graph)

tf.global_variables_initializer().run()
for step in range(EPOCHS):
    length = len(dataset)
    for i in range(length/batch_size):
        if step <= 5:
            batch_xs = utils.next_batch(dataset,i,batch_size, ctr)
            _,_,_ = sess.run([d_optim,e_optim, dis_optim], feed_dict={x: batch_xs, lr : 1e-3})
            reconstructions_, latents_ = sess.run([reconstructions,latents], feed_dict={x: batch_xs, lr : 1e-3}) 
            enc_loss = sess.run(full_enc_cost, feed_dict={x: batch_xs, lr : 1e-3})
            dec_loss = sess.run(dec_cost, feed_dict={x:batch_xs, lr : 1e-3})
            discrim_loss = sess.run(discrim_cost, feed_dict={x:batch_xs, lr : 1e-3})
            total_loss = enc_loss + dec_loss + discrim_loss
            loss_dict = [total_loss, enc_loss, dec_loss, discrim_loss]
        elif step<=8:
            batch_xs = utils.next_batch(dataset,i,batch_size, ctr)
            _,_,_ = sess.run([d_optim,e_optim, dis_optim], feed_dict={x: batch_xs, lr : 1e-4})
            reconstructions_, latents_ = sess.run([reconstructions,latents], feed_dict={x: batch_xs,lr : 1e-4})
            enc_loss = sess.run(full_enc_cost, feed_dict={x: batch_xs,lr : 1e-4})
            dec_loss = sess.run(dec_cost, feed_dict={x:batch_xs, lr : 1e-4})
            discrim_loss = sess.run(discrim_cost, feed_dict={x:batch_xs, lr : 1e-4})
            total_loss = enc_loss + dec_loss + discrim_loss
            loss_dict = [total_loss, enc_loss, dec_loss, discrim_loss]
    	elif step <=10:
       	    batch_xs = utils.next_batch(dataset,i,batch_size, ctr)
            _,_,_ = sess.run([d_optim,e_optim, dis_optim], feed_dict={x: batch_xs, lr : 5e-5})
            reconstructions_, latents_ = sess.run([reconstructions,latents], feed_dict={x: batch_xs, lr : 5e-5})
            enc_loss = sess.run(full_enc_cost, feed_dict={x: batch_xs, lr : 5e-5})
            dec_loss = sess.run(dec_cost, feed_dict={x:batch_xs, lr : 5e-5})
            discrim_loss = sess.run(discrim_cost, feed_dict={x:batch_xs, lr : 5e-5})
            total_loss = enc_loss + dec_loss + discrim_loss
            loss_dict = [total_loss, enc_loss, dec_loss, discrim_loss]
        if i%100 == 0:
            if step<=5:
               print "Learning rate is {}".format(1e-3)
    	    elif step <=8:
    	       print "Learning rate is {}".format(1e-4)
    	    elif step <=10:
	           print "Learning rate is {}".format(5e-5)
            print "Epoch = {}, Iteration = {}".format(step, i)
            print "Total Loss = {0}, Encoder loss = {1}, Decoder Loss = {2}, Discriminator Loss = {3}".format(*loss_dict)
            #summary = sess.run(merge, feed_dict={x:batch_x})
            #writer.add_summary(summary, i)
        if i%200 == 0:
            test_image = utils.next_image(dataset, i,batch_size)
            ctr = ctr+1
            sample_test = sess.run(reconstructions, feed_dict = {x : test_image, lr : 1e-3})
            test_image.astype(np.float32)
    	    sample_test = np.squeeze(sample_test)
    	    test_image = np.squeeze(test_image)
    	    scipy.misc.imsave("./output/imgtest_epoch_{}_iteration_{}.bmp".format(step, i),sample_test)
    	    scipy.misc.imsave("./output/imgorg1_epoch_{}_iteration_{}.bmp".format(step, i),test_image )

