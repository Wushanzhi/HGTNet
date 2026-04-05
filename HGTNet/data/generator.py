# -*- coding: utf-8 -*-
import numpy as np
from utils.io_utils import readTif


def train_generator(img_file_dst, lbl_file_dst, batch_size, shuffle):
    idx = np.arange(len(img_file_dst))

    if shuffle:
        np.random.shuffle(idx)
    total_data_num = len(img_file_dst)

    while True:
        for i in range(total_data_num // batch_size):
            max_ = min((i + 1) * batch_size, total_data_num)
            tmp_file = idx[i * batch_size:max_]

            img_data = np.ndarray([max_ - i * batch_size, 256, 256, 24])
            lbl_data = np.zeros([max_ - i * batch_size, 256, 256, 1])

            for j, tmp_idx in enumerate(tmp_file):
                tmp_img = readTif(img_file_dst[tmp_idx])
                tmp_lbl = readTif(lbl_file_dst[tmp_idx])

                img_data[j] = tmp_img.transpose(1, 2, 0)
                lbl_data[j, :, :, 0] = tmp_lbl.squeeze()

            yield (img_data, lbl_data)