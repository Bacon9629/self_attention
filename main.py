import numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

from SelfAttention import SelfAttention1D
from CHATGPT_SelfAttention import SelfAttention

        
if __name__ == "__main__":
    x = np.zeros((34045, 2250), dtype="float32")
    y = np.zeros((34045, 12), dtype="float32")

    inp = Input(shape=[x.shape[1], ])
    # att = SelfAttention(1, x.shape[1])
    att = SelfAttention1D()
    dense1 = Dense(512)
    batch_norm1 = BatchNormalization()
    dense2 = Dense(128)
    batch_norm2 = BatchNormalization()
    out = Dense(y.shape[1], activation='softmax')

    z = att(inp)
    z = Flatten()(z)
    z = dense1(z)
    z = LeakyReLU()(z)
    z = batch_norm1(z)
    z = dense2(z)
    z = LeakyReLU()(z)
    z = batch_norm2(z)
    z = out(z)

    model = tf.keras.Model(inputs=inp, outputs=z)
    model.build(x.shape)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('best_model_self_att.weight', monitor='val_accuracy', save_weights_only=True, save_best_only=True, mode='max', verbose=1)
    model.fit(x, y, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[checkpoint])