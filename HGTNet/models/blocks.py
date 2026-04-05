# -*- coding: utf-8 -*-
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Add


# 定义残差块
def residual_block(x, filters, kernel_size=3):
    shortcut = Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_normal")(x)  # 1×1 卷积匹配通道数

    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    # 添加残差连接（输入 + 卷积输出）
    x = Add()([shortcut, x])
    x = Activation("relu")(x)

    return x