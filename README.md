# Convolutional_Adversarial_Autoencoder
An Adversarial Autoencoder with a Deep Convolutional Encoder and Decoder Network

Modification of the Adversarial Autoencoder which uses the Generative Adversarial Networks(GAN) to perform variational inference by matching the aggregated posterior of the encoder with an arbitrary prior distribution. This is the modified implementation of the paper Adversarial Autoencoders(https://arxiv.org/abs/1511.05644) by Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey.

Adversarial Autoencoder learns to shirk an image to a latent dimension and reconstruct the same image. In this process the network learns to represent each image in latent dimension. This latent representation can then be used as the unique identifier of the image. We can then perform various operations like clustering, labelling etc for the whole dataset. This is an unsupervised learning technique through which we can label large amounts of unlabelled data. And then further carry out the supervised learning techniques.

In this code the network is trained on Coco dataset of 87,000 images.

However,according to the experimental results a fully connected encoder decoder network performs better than the convolutional encoder decoder network.
