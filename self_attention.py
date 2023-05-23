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
        print("SelfAttention block input shape", inputs.shape)
        print("SelfAttention block output shape", output.shape)

        ## output = tf.reshape(output, (-1, inputs.shape[1]))  ## 我改成攤平了，沒有這個才正常


        return output
        
        
if __name__ == "__main__":
    x = np.zeros((34045, 2250), dtype="float32")
    y = np.zeros((34045, 12), dtype="float32")

    inp = Input(shape=[x.shape[1], ])
    att = SelfAttention(1, x.shape[1])
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
    model.fit(x, y, epochs=50, batch_size=batch_size, verbose=1, validation_split=0.2, callbacks=[checkpoint])