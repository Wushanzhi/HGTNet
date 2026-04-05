# -*- coding: utf-8 -*-
import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from models.hgtnet import HGTnet
from data.generator import train_generator


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    batch_size = 4
    model = HGTnet()

    train_image_dst = glob.glob(r'G:\center pivot\all\clip2\img\train\*.tif')
    train_label_dst = glob.glob(r'G:\center pivot\all\clip2\lab\train_50_1\*.tif')
    shuffle = True
    train_Generator = train_generator(
        train_image_dst,
        train_label_dst,
        batch_size,
        shuffle)

    val_image_dst = glob.glob(r'G:\center pivot\all\clip2\img\val\*.tif')
    val_label_dst = glob.glob(r'G:\center pivot\all\clip2\lab\val_50_1\*.tif')
    val_Generator = train_generator(
        val_image_dst,
        val_label_dst,
        batch_size,
        shuffle)

    steps_per_epoch = len(train_image_dst) / batch_size
    val_steps = len(val_image_dst) / batch_size
    model_path = r"G:\center pivot\all\0831\model.hdf5"  # 模型存储地址

    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='loss',
        verbose=1,  # 日志显示模式:0->安静模式,1->进度条,2->每轮一行
        save_best_only=True
    )

    history = model.fit_generator(
        train_Generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        callbacks=[model_checkpoint],  # 实时保存最佳模型
        validation_data=val_Generator,
        validation_steps=val_steps
    )

    model.save(r'G:\center pivot\all\0831\model.h5')  # 保存最终模型