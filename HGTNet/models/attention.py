# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Multiply, Conv2D, Concatenate, SeparableConv2D, Add, Lambda


def EG_attention(input_tensor):
    channels = K.int_shape(input_tensor)[-1]
    height, width = K.int_shape(input_tensor)[1], K.int_shape(input_tensor)[2]
    min_dim = min(height, width)
    if min_dim >= 128:
        dilation_rate = 4
    elif min_dim >= 64:
        dilation_rate = 2
    else:
        dilation_rate = 1

    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])  # shape: (batch, H, W, 2)

    # Sobel 边缘引导
    def multi_channel_sobel(x):
        # x: [batch, H, W, channels]
        sobel = tf.image.sobel_edges(x)  # shape: (B, H, W, C, 2)
        edge_mag = tf.sqrt(tf.square(sobel[..., 0]) + tf.square(sobel[..., 1]) + 1e-6)  # shape: (B, H, W, C)
        return tf.reduce_mean(edge_mag, axis=-1, keepdims=True)  # shape: (B, H, W, 1)

    edge_mag = Lambda(multi_channel_sobel)(input_tensor)
    edge_feat = Conv2D(8, 3, padding='same', activation='relu')(edge_mag)
    edge_weight = Conv2D(1, 1, padding='same', activation='sigmoid')(edge_feat)
    # 反向注意力
    reverse_weight = Lambda(lambda x: 1.0 - x)(edge_weight)

    # 注意力主干结构
    conv = SeparableConv2D(channels, (3, 3), dilation_rate=dilation_rate, padding="same",
                           activation="relu",
                           depthwise_initializer="he_normal",
                           pointwise_initializer="he_normal")(concat)

    conv2 = SeparableConv2D(channels, (3, 3), dilation_rate=dilation_rate, padding="same",
                            activation="relu",
                            depthwise_initializer="he_normal",
                            pointwise_initializer="he_normal")(conv)

    attention_raw = Conv2D(1, 1, padding="same", activation="sigmoid")(conv2)

    edge_enhance = Multiply()([input_tensor, edge_weight])
    background_suppress = Multiply()([input_tensor, reverse_weight])

    # 融合 shape attention 主干与边缘注意力和反向注意力
    attention = Add()([
        Multiply()([input_tensor, attention_raw]),
        0.5 * edge_enhance,
        -0.5 * background_suppress
    ])
    return Add()([input_tensor, attention])


def MS_attention(input_tensor):
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)

    channels = K.int_shape(input_tensor)[-1]
    height, width = K.int_shape(input_tensor)[1], K.int_shape(input_tensor)[2]
    min_dim = min(height, width)
    if min_dim >= 128:
        dilation_rate = 4
    elif min_dim >= 64:
        dilation_rate = 2
    else:
        dilation_rate = 1

    # 拼接池化结果，形状为 (batch, H, W, 2)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    # 使用空洞卷积来扩展感受野
    conv = SeparableConv2D(channels, (3, 3), dilation_rate=dilation_rate, padding="same",
                           activation="relu",
                           depthwise_initializer="he_normal",
                           pointwise_initializer="he_normal")(concat)

    # 通过多层卷积提取更细粒度的空间信息
    conv2 = SeparableConv2D(channels, (3, 3), dilation_rate=dilation_rate, padding="same",
                            activation="relu",
                            depthwise_initializer="he_normal",
                            pointwise_initializer="he_normal")(conv)

    attention = Conv2D(1, (1, 1), padding="same",
                       activation="sigmoid",
                       kernel_initializer="he_normal")(conv2)

    return Multiply()([input_tensor, attention])