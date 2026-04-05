# -*- coding: utf-8 -*-
import tensorflow.keras.backend as K


def focal_loss(alpha=0.25, gamma=1.5):
    def focal_loss_fixed(y_true, y_pred):
        # 确保数值稳定性（防止 log(0)）
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # 计算交叉熵
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

        # 计算 p_t（真实类别的概率）
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # 计算调制因子 (1 - p_t)^gamma
        modulating_factor = K.pow(1 - p_t, gamma)

        # 应用 alpha 权重（正负样本不同）
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)

        # 组合所有因子
        loss = alpha_weight * modulating_factor * cross_entropy

        # 返回均值损失
        return K.mean(loss, axis=-1)

    return focal_loss_fixed