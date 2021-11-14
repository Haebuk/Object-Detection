import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Resizing, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Input, LayerNormalization, Dense, GlobalAveragePooling1D, Dropout
from keras import layers

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images): # keras layer computation
        batch_size = tf.shape(images)[0]
        # https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
        patches = tf.images.extract_patches( 
            images=images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID',
        )
        dim = patches.shape[-1]
        patches = tf.reshape(patches, (batch_size, -1, dim))
        return patches

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, S, C, Ds, Dc):
        """
        S: number of patches
        C: hidden dimensions
        Ds: tunnable hidden widths in the token-mixing
        Dc: tunnable hidden widths in the channel-mixing
        """
        super(MLPBlock, self).__init__()
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        init_w = tf.random_normal_initializer()
        self.DS, self.Dc = Ds, Dc
        self.W1 = tf.Variable(
            initial_value=init_w(shape=(S, Ds), dtype=tf.float32),
            trainable=True,
        )
        self.W2 = tf.Variable(
            initial_value=init_w(shape=(Ds, S), dtype=tf.float32),
            trainable=True,
        )
        self.W3 = tf.Variable(
            initial_value=init_w(shape=(C, Dc), dtype=tf.float32),
            trainable=True,
        )
        self.W4 = tf.Variable(
            initial_value=init_w(shape=(Dc, C), dtype=tf.float32),
            trainable=True,
        )

    def call(self, X):
        # Patches
        # (batch_size, S, C)
        batch_size, S, C = X.shape

        # Token-Mixing
        # (batch_size, C, S)
        X_T = tf.transpose(self.layer_norm1(X), perm=[0, 2, 1])
        W1X = tf.matmul(X_T, self.W1) # (batch_size, C, S) * (S, Ds) = (batch_size, C, Ds)
        gelu_W1X = tf.nn.gelu(W1X) # (batch_size, C, Ds)
        # (batch_size, C, Ds) * (Ds, S) = (batch_size, C, S) -> gelu_W1X * W2
        # (S, C) + (batch_size, S, C)^T = (batch_size, S, C) -> X + (gelu_W1X * W2)^T = U
        U = X + tf.transpose(tf.matmul(gelu_W1X, self.W2), perm=[0, 2, 1])
        
        # Channel-Mixing
        # U = (batch_size, S, C)
        # W3U = U * W3 : (batch_size, S, C) * (C, Dc) = (batch_size, S, Dc)
        W3U = tf.matmul(U, self.W3) # (batch_size, S, Dc)
        gelu_W3U = tf.nn.gelu(W3U) # (batch_size, S, Dc)
        # W4 = (Dc, C)
        Y = U + tf.matmul(gelu_W3U, self.W4) # (batch_size, S, C)

        return Y

class MLPMixer(tf.keras.models.Model):
    def __init__(self, patch_size, S, C, DS, DC, num_of_mlp_blocks, image_size, batch_size, num_classes):
        super(MLPMixer, self).__init__()
        self.projection = Dense(C)
        self.mlpBlocks = [MLPBlock(S, C, DS, DC) for _ in range(num_of_mlp_blocks)]
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.S = S
        self.C = C
        self.DS = DS
        self.DC = DC
        self.image_size = image_size
        self.num_classes = num_classes

        self.data_augmentation = tf.keras.Sequential(
            [
                Normalization(),
                RandomFlip('horizontal'),
                RandomRotation(0.02),
                RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name = 'data_augmentation',
        )

        self.classificationLayer = Sequential([
            GlobalAveragePooling1D(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

    def extract_patches(self, images, patch_size):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )

        patches = tf.reshape(patches, [batch_size, -1, 3 * patch_size ** 2])

        return patches

    def call(self, images):
        # input image shape: (batch_size, image_size, image_size, 3) = (3, 64, 64, 3)

        batch_size = images.shape[0]

        augmented_images = self.data_augmentation(images)

        # patches shape: (batch_size, S, 3 * patch_size ** 2)
        X = self.extract_patches(augmented_images, self.patch_size)

        # Per-patch fully-connected
        # X shape: (batch_size, S, C)
        X = self.projection(X)

        for block in self.mlpBlocks:
            X = block(X)

        out = self.classificationLayer(X)

        return out