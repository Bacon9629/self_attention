import numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint


class SelfAttention1D(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def get_config(self):
        config = super().get_config()
        config.update({
            "Wq": self.Wq,
            "Wk": self.Wk,
            "Wv": self.Wv,
        })
        return config

    def build(self, input_shape):
        self.Wq = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', name='Wq', trainable=True)
        self.Wk = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', name='Wq', trainable=True)
        self.Wv = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', name='Wq', trainable=True)
        pass

    def call(self, inputs, *args, **kwargs):
        q = tf.matmul(inputs, self.Wq)
        k = tf.matmul(inputs, self.Wk)
        v = tf.matmul(inputs, self.Wv)

        q = tf.expand_dims(q, -1)
        k = tf.expand_dims(k, -2)

        s_ = tf.keras.backend.batch_dot(q, k)

        s = Softmax()(s_)
        outs = tf.keras.backend.batch_dot(s, v)

        return outs


if __name__ == '__main__':
    x = np.random.random(200).reshape([3, 100])
    y = np.random.random(6).reshape([2, 3])

    inputs = Input([100])
    outs = SelfAttention1D()(inputs)
    outs = Dense(3, activation="sigmoid")(outs)

    model = Model(inputs=inputs, outputs=outs)

    model.summary()
    model.compile(optimizer="adam", loss=tf.losses.MSE, metrics="accuracy")

    model.fit(x, y, epochs=20, verbose=2)




