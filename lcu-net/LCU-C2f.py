import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Concatenate, BatchNormalization, Activation

class Bottleneck(Layer):
    def __init__(self, filters, kernel_size=(3, 3), shortcut=True):
        super(Bottleneck, self).__init__()
        self.conv = Conv2D(filters, kernel_size, padding='same')
        self.bn = BatchNormalization()
        self.shortcut = shortcut

    def call(self, x):
        y = self.bn(self.conv(x))
        if self.shortcut:
            y += x
        return y

class C2f(Layer):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(C2f, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv2D(2 * self.c, kernel_size=1, strides=1, padding='same')
        self.cv2 = Conv2D(c2, kernel_size=1, strides=1, padding='same')
        self.m = [Bottleneck(self.c, kernel_size=(3, 3), shortcut=shortcut) for _ in range(n)]

    def call(self, x):
        y = tf.split(self.cv1(x), num_or_size_splits=2, axis=3)
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(tf.concat(y, axis=3))


def __encoder_decoder_block(input_tensor, c1, c2, n=1, encoder=True, concat_block=None, dropout=0.0,
                            maxpool=True, debug=False):
    global block_num

    if debug:
        print("Block number:", block_num)
        print('Input tensor of shape: ', input_tensor.get_shape())

    c2f_layer = C2f(c1, c2, n=n)
    conv_output = c2f_layer(input_tensor)

    # Lanjutkan dengan sisa blok encoder atau decoder
    if not encoder:
        assert concat_block is not None
        merged = Concatenate(axis=-1)([concat_block, conv_output])
        conv_output = merged

    if dropout > 0.0 and encoder:
        conv_output = Dropout(dropout)(conv_output)
    if maxpool and encoder:
        conv_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_output)

    if debug:
        print('Returning tensor of shape:', conv_output.get_shape())
        block_num += 1

    return conv_output, conv_output

def LCU_Net_with_C2f(input_shape=(256, 256, 3), output_shape=(256, 256, 1), debug=False):
    input_tensor = Input(input_shape)
    input_tensor = BatchNormalization()(input_tensor)

    # Menggunakan __encoder_decoder_block dengan C2f
    block1, down_sampled1 = __encoder_decoder_block(input_tensor=input_tensor, c1=3, c2=16, encoder=True, dropout=0.0, maxpool=True, debug=debug)
    block2, down_sampled2 = __encoder_decoder_block(input_tensor=down_sampled1, c1=16, c2=32, encoder=True, dropout=0.0, maxpool=True, debug=debug)

    # Encoder Blocks with C2f
    block3, down_sampled3 = __encoder_decoder_block(input_tensor=down_sampled2, c1=32, c2=64, encoder=True, dropout=0.0, maxpool=True, debug=debug)
    block4, down_sampled4 = __encoder_decoder_block(input_tensor=down_sampled3, c1=64, c2=128, encoder=True, dropout=0.0, maxpool=True, debug=debug)
    block5, down_sampled5 = __encoder_decoder_block(input_tensor=down_sampled4, c1=128, c2=256, encoder=True, dropout=0.0, maxpool=True, debug=debug)

    # Decoder Blocks with C2f
    block6, _ = __encoder_decoder_block(input_tensor=down_sampled5, c1=256, c2=128, encoder=False, concat_block=block4, debug=debug)
    block7, _ = __encoder_decoder_block(input_tensor=block6, c1=128, c2=64, encoder=False, concat_block=block3, debug=debug)
    block8, _ = __encoder_decoder_block(input_tensor=block7, c1=64, c2=32, encoder=False, concat_block=block2, debug=debug)
    block9, _ = __encoder_decoder_block(input_tensor=block8, c1=32, c2=16, encoder=False, concat_block=block1, debug=debug)

    # Final Convolutional Layer
    output_block = Conv2D(filters=output_shape[-1], kernel_size=1, strides=1, padding='same', activation='sigmoid')(block9)

    lcu_net = tf.keras.Model(inputs=input_tensor, outputs=output_block)
    lcu_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if debug:
        lcu_net.summary()

    return lcu_net

# Membuat model
model = LCU_Net_with_C2f(debug=True)

