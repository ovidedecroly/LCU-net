#################################################
# @Author Aditya Chakma
# @Github https://github.com/Aitto
#################################################

# Added in case dlls are not found for GPU
import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow_addons as tfa


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
                     activation='relu',
                     kernel_initializer='he_normal')(input_tensor)
    conv3x1 = BatchNormalization()(conv3x1)

    conv1x3 = Conv2D(filters=filter_size,
                     kernel_size=(1, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     kernel_initializer='he_normal')(conv3x1)
    conv1x3 = BatchNormalization()(conv1x3)

    return conv1x3


block_num = 1


def __encoder_decoder_block(input_tensor, filter_size: int, encoder: bool, concat_block=None, dropout=0.0,
                            maxpool: bool = True, debug: bool = False):
    """
    Creates encoder and decoder block
    :param input_tensor: The input tensor
    :param filter_size: Filter size to be used in Convolution layers
    :param encoder: Boolean. Create encoder or decoder block
    :param concat_block: Must pass if encoder=False. The block to concat with input tensor after transformation
    :param dropout: Amount of dropout to use. Default is 0.0
    :param maxpool: Boolean. Use MaxPooling layer or not. Default is troo.
    :param debug: Boolean. Print debugging messages or not.
    :return: output block, output block after using MaxPooling
    """

    global block_num

    if debug:
        print("Block number:", block_num)
        print('Input tensor of shape: ', input_tensor.get_shape())

    if not encoder:
        up_sampled = Conv2DTranspose(filters=filter_size,
                                     kernel_size=(2, 2),
                                     strides=(2, 2),
                                     padding='same',
                                     kernel_initializer='he_normal')(input_tensor)

        assert concat_block is not None

        if debug:
            print('Up sampled tensor of shape:', up_sampled.get_shape())
            print('Concat tensor of shape', concat_block.get_shape())

        merged = Concatenate(axis=-1)([concat_block, up_sampled])

        if debug:
            print('Merged tensor of shape:', merged.get_shape())

        input_tensor = merged

    conv1 = Conv2D(filters=filter_size,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='he_normal')(input_tensor)
    conv1 = BatchNormalization()(conv1)

    conv3 = __apply_conv3x3(input_tensor, filter_size)

    conv5 = __apply_conv3x3(conv3, filter_size)

    conv7 = __apply_conv3x3(conv5, filter_size)

    merged = Concatenate(axis=-1)([conv1, conv3, conv5, conv7])
    # merged = BatchNormalization()(merged)

    block = merged

    if dropout > 0.0 and encoder:
        merged = Dropout(dropout)(merged)
    if maxpool and encoder:
        merged = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(merged)

    down_sampled = merged

    if debug:
        print('Returning tensor of shape:', merged.get_shape())
        block_num += 1

    return block, down_sampled


def LCU_Net(input_shape=(256, 256, 3), output_shape=(256, 256, 1), debug: bool = False):
    """
    Create LCU-Net model
    :param input_shape: shape of input image. Default shape is (256, 256, 3)
    :param output_shape: output shape of the model. Default shape is (256, 256, 1)
    :param debug: print debug messages or not. Default is False. Used for trial and error.
    :return: returns the LCU-Net model. (Made with tensorflow)
    """

    dropout = 0.0

    input_tensor = Input(input_shape)
    input_tensor = BatchNormalization()(input_tensor)

    block1, down_sampled1 = __encoder_decoder_block(input_tensor=input_tensor,
                                                    filter_size=16,
                                                    encoder=True,
                                                    dropout=dropout,
                                                    maxpool=True,
                                                    debug=debug)
    block2, down_sampled2 = __encoder_decoder_block(input_tensor=down_sampled1,
                                                    filter_size=32,
                                                    encoder=True,
                                                    dropout=dropout,
                                                    maxpool=True,
                                                    debug=debug)
    block3, down_sampled3 = __encoder_decoder_block(input_tensor=down_sampled2,
                                                    filter_size=64,
                                                    encoder=True,
                                                    dropout=dropout,
                                                    maxpool=True,
                                                    debug=debug)
    block4, down_sampled4 = __encoder_decoder_block(input_tensor=down_sampled3,
                                                    filter_size=128,
                                                    encoder=True,
                                                    dropout=dropout,
                                                    maxpool=True,
                                                    debug=debug)
    block5, down_sampled5 = __encoder_decoder_block(input_tensor=down_sampled4,
                                                    filter_size=256,
                                                    encoder=True,
                                                    dropout=dropout,
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

    block10 = Conv2D(filters=16,
                     kernel_size=(3, 1),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     kernel_initializer='he_normal')(block9)
    block10 = Conv2D(filters=16,
                     kernel_size=(1, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     kernel_initializer='he_normal')(block10)

    filters = output_shape[2] if len(output_shape) == 3 else 1
    output_block = Conv2D(filters=filters,
                          kernel_size=1,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal')(block10)

    # output_block = tfa.layers.CRF(4)(output_block)

    lcu_net = tf.keras.Model(inputs=input_tensor, outputs=output_block)
    lcu_net.compile(optimizer=keras.optimizers.Adam(learning_rate=1.5e-4),
                    loss='binary_crossentropy',
                    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    if debug:
        lcu_net.summary()

    tf.keras.utils.plot_model(lcu_net)

    print("Model successfully made!")

    return lcu_net


if __name__ == "__main__":
    model = LCU_Net(debug=True)

