import numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint


class SelfAttention(Layer):
    def __init__(self, num_heads, key_dim):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "Wq": self.Wq,
            "Wk": self.Wk,
            "Wv": self.Wv,
        })
        return config

    def build(self, input_shape):
        self.Wq = self.add_weight(shape=(input_shape[-1], self.key_dim), initializer='glorot_uniform', name='Wq')
        self.Wk = self.add_weight(shape=(input_shape[-1], self.key_dim), initializer='glorot_uniform', name='Wk')
        self.Wv = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', name='Wv')

    def call(self, inputs):
        q = tf.matmul(inputs, self.Wq)
        k = tf.matmul(inputs, self.Wk)
        v = tf.matmul(inputs, self.Wv)

        q = tf.reshape(q, (-1, self.num_heads, self.key_dim))
        k = tf.reshape(k, (-1, self.num_heads, self.key_dim))
        v = tf.reshape(v, (-1, self.num_heads, inputs.shape[-1]))

        scores = tf.matmul(q, k, transpose_b=True)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, v)
        print(inputs.shape)
        print(output.shape)

        output = tf.reshape(output, (-1, inputs.shape[1]))  ## 我改成攤平了，沒有這個才正常


        return output

