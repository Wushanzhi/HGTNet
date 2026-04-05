# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, ConvLSTM2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D

from models.attention import EG_attention, MS_attention
from models.blocks import residual_block
from losses.focal_loss import focal_loss


# keras函数式建模
def HGTnet(batch_size=4, input_size=(256, 256, 24), classNum=1, learning_rate=1e-4):
    inputs = Input(input_size)  # 输入的图像大小（行，列，波段数）
    BS = batch_size

    def reshapes(embed):
        # => [BS, 256, 256, 4, 6]
        embed = tf.reshape(embed, [BS, 256, 256, 4, 6])
        # => [BS, 256, 256, 6, 4]
        embed = tf.transpose(embed, [0, 4, 1, 2, 3])
        return embed

    inputs1 = keras.layers.Lambda(reshapes)(inputs)
    conv_lstm1 = ConvLSTM2D(filters=64,
                            kernel_size=(3, 3),
                            padding="same",
                            return_sequences=False)(inputs1)

    # 编码器
    conv1 = residual_block(conv_lstm1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = residual_block(pool3, 512)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = residual_block(pool4, 1024)
    drop5 = Dropout(0.5)(conv5)

    # 解码器
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6))
    conv6 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6))

    conv3_att = EG_attention(conv3)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3_att, up7], axis=3)
    conv7 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7))
    conv7 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7))

    conv2_att = MS_attention(conv2)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2_att, up8], axis=3)
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8))
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8))

    conv1_att = MS_attention(conv1)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1_att, up9], axis=3)
    conv9 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9))
    conv9 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(classNum, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss=focal_loss(alpha=0.25, gamma=1.5),
        metrics=['Precision', 'Recall']
    )

    return model