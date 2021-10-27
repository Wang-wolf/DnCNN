import os
import cv2
import h5py
import glob
import random
import numpy as np
import torch
import torch.utils.data as udata


def image2patch(img, win, stride=1):
    k = 0
    img_c, img_w, img_h = img.shape
    # 按照stride的大小進行像素的擷取
    patch = img[:, 0:img_w-win+1:stride, 0:img_h-win+1:stride]
    total_patch_num = patch.shape[1] * patch.shape[2]
    patches = np.zeros([img_c, win*win, total_patch_num], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:img_w-win+i+1:stride, j:img_h-win+j+1:stride]
            patches[:, k, :] = np.array(patch[:]).reshape(img_c, total_patch_num)
            k += 1

    return patches.reshape([img_c, win, win, total_patch_num])


def data_augmentation(image, mode):
    # 將維度轉成(w, h, c)
    output = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # 不進行處理
        output = output
    elif mode == 1:
        # 上下翻轉
        output = np.flipud(output)
    elif mode == 2:
        # 順時鐘轉90度
        output = np.rot90(output)
    elif mode == 3:
        # 順時鐘轉90度、上下翻轉
        output = np.rot90(output)
        output = np.flipud(output)
    elif mode == 4:
        # 順時鐘轉180度、上下翻轉
        output = np.rot90(output, k=2)
    elif mode == 5:
        # 順時鐘轉180度、上下翻轉
        output = np.rot90(output, k=2)
        output = np.flipud(output)
    elif mode == 6:
        # 順時鐘轉270度、上下翻轉
        output = np.rot90(output, k=3)
    elif mode == 7:
        # 順時鐘轉270度、上下翻轉
        output = np.rot90(output, k=3)
        output = np.flipud(output)

    # 將維度轉回(c, w, h)
    return np.transpose(output, (2, 0, 1))


def prepare_data(data_path, patch_size, stride, aug_times=1):
    print('處理訓練資料中')
    scales = [1, 0.9, 0.8, 0.7]
    # 訓練集路徑
    train_data_path = os.path.join(data_path, 'train', '*.png')
    # 獲取所有影像路徑
    files = glob.glob(train_data_path)
    files.sort()
    # 建立h5檔以進行資料的儲存
    h5_file = h5py.File('train.h5py', 'w')

    # 訓練集
    train_num = 0
    for i in range(len(files)):
        print('處理中...第%d張影像' % (i + 1))
        # 讀取圖片
        image = cv2.imread(files[i])
        h, w, c = image.shape

        for k in range(len(scales)):
            img = cv2.resize(image, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img[:, :, 0].copy(), 0)
            img = np.float32(img/255.0)
            patches = image2patch(img, win=patch_size, stride=stride)
            print("檔案:%s 縮放比例:%.1f 採樣像素:%d" % (files[i], scales[k], patches.shape[3] * aug_times))

            # 儲存原始影像
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                h5_file.create_dataset(str(train_num), data=data)
                train_num += 1

                # 儲存經過資料增強的影像
                for m in range(aug_times-1):
                    # 隨機使用一種方式進行資料增強
                    data_aug = data_augmentation(data, np.random.randint(0, 8))
                    h5_file.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5_file.close()

    # 驗證集
    print('\n處理驗證資料中')
    # 驗證集路徑
    val_data_path = os.path.join(data_path, 'Set12', '*.png')
    # 獲取所有影像路徑
    files = glob.glob(val_data_path)
    files.sort()
    h5_file = h5py.File('val.h5py', 'w')

    # 前處理
    val_num = 0
    for i in range(len(files)):
        print("檔案:%s" % (files[i]))
        image = cv2.imread(files[i])
        # 在0的位置新增維度
        image = np.expand_dims(image[:, :, 0], 0)
        image = np.float32(image/255.0)  # 正規化
        h5_file.create_dataset(str(val_num), data=image)
        val_num += 1
    h5_file.close()
    print('訓練集, 採樣數量: %d\n' % train_num)
    print('驗證集, 採樣數量: %d\n' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5_file = h5py.File('train.h5py', 'r')
        else:
            h5_file = h5py.File('val.h5py', 'r')
        self.keys = list(h5_file.keys())
        random.shuffle(self.keys)
        h5_file.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5_file = h5py.File('train.h5py', 'r')
        else:
            h5_file = h5py.File('val.h5py', 'r')
        key = self.keys[index]
        data = np.array(h5_file[key])
        h5_file.close()
        return torch.Tensor(data)
