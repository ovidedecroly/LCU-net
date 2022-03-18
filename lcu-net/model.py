import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


def __apply_conv3x3(input_tensor, filter_size: int):
    """
    Apply 3x3 conv to input conv with corresponding filter size
    :param input_tensor: the input tensor to apply 3x3 convolution
    :param filter_size: filter size for the convolution operation
    :return: tensor after applying convolution
    """

    conv3x1 = Conv2D(filters=filter_size,
                     kernel_size=(3, 1),
                     strides=(1, 1),
                     padding='same',
                     activation='relu')(input_tensor)
    conv3x1 = BatchNormalization()(conv3x1)

    conv1x3 = Conv2D(filters=filter_size,
                     kernel_size=(1, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu')(conv3x1)
    conv1x3 = BatchNormalization()(conv1x3)

    return conv1x3


block_num = 1


def __encoder_decoder_block(input_tensor, filter_size: int, encoder: bool, concat_block=None, dropout=0.5,
                            maxpool: bool = True, debug: bool = False):

    global block_num

    if debug:
        print("Block number:", block_num)
        print('Input tensor of shape: ', input_tensor.get_shape())

    if not encoder:
        up_sampled = Conv2DTranspose(filters=filter_size,
                                     kernel_size=(2, 2),
                                     strides=(2, 2),
                                     padding='same')(input_tensor)

        assert concat_block is not None

        if debug:
            print('Up sampled tensor of shape:', up_sampled.get_shape())
            print('Concat tensor of shape', concat_block.get_shape())

        merged = Concatenate(axis=-1)([concat_block, up_sampled])

        if debug:
            print('Merged tensor of shape:', merged.get_shape())

        input_tensor = merged

    conv1 = Conv2D(filters=filter_size, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    conv1 = BatchNormalization()(conv1)

    conv3 = __apply_conv3x3(input_tensor, filter_size)

    conv5 = __apply_conv3x3(conv3, filter_size)

    conv7 = __apply_conv3x3(conv5, filter_size)

    merged = Concatenate(axis=-1)([conv1, conv3, conv5, conv7])

    block = merged

    if dropout > 0 and encoder:
        merged = Dropout(dropout)(merged)
    if maxpool and encoder:
        merged = MaxPooling2D(pool_size=(2, 2))(merged)

    down_sampled = merged

    if debug:
        print('Returning tensor of shape:', merged.get_shape())
        block_num += 1

    return block, down_sampled


def LCU_Net(input_shape=(256, 256, 3), debug: bool = False):
    input_tensor = Input(input_shape)

    block1, down_sampled1 = __encoder_decoder_block(input_tensor=input_tensor,
                                                    filter_size=16,
                                                    encoder=True,
                                                    dropout=0.8,
                                                    maxpool=True,
                                                    debug=debug)
    block2, down_sampled2 = __encoder_decoder_block(input_tensor=down_sampled1,
                                                    filter_size=32,
                                                    encoder=True,
                                                    dropout=0.8,
                                                    maxpool=True,
                                                    debug=debug)
    block3, down_sampled3 = __encoder_decoder_block(input_tensor=down_sampled2,
                                                    filter_size=64,
                                                    encoder=True,
                                                    dropout=0.8,
                                                    maxpool=True,
                                                    debug=debug)
    block4, down_sampled4 = __encoder_decoder_block(input_tensor=down_sampled3,
                                                    filter_size=128,
                                                    encoder=True,
                                                    dropout=0.8,
                                                    maxpool=True,
                                                    debug=debug)
    block5, down_sampled5 = __encoder_decoder_block(input_tensor=down_sampled4,
                                                    filter_size=256,
                                                    encoder=True,
                                                    dropout=0.8,
                                                    maxpool=True,
                                                    debug=debug)

    block6, _ = __encoder_decoder_block(input_tensor=block5,
                                        filter_size=128,
                                        encoder=False,
                                        concat_block=block4,
                                        debug=debug)
    block7, _ = __encoder_decoder_block(input_tensor=block6,
                                        filter_size=64,
                                        encoder=False,
                                        concat_block=block3,
                                        debug=debug)
    block8, _ = __encoder_decoder_block(input_tensor=block7,
                                        filter_size=32,
                                        encoder=False,
                                        concat_block=block2,
                                        debug=debug)
    block9, _ = __encoder_decoder_block(input_tensor=block8,
                                        filter_size=16,
                                        encoder=False,
                                        concat_block=block1,
                                        debug=debug)

    filters = input_shape[2] if len(input_shape) == 3 else 1
    output_block = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', activation='sigmoid')(block9)

    lcu_net = tf.keras.Model(inputs=input_tensor, outputs=output_block)
    lcu_net.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if debug:
        model.summary()

    print("Model successfully made!")

    return lcu_net


model = LCU_Net(debug=False)
