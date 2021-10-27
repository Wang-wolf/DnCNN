import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from models import DnCNN
from utils import batch_psnr


def main():
    # 設置參數
    num_of_layers = 17

    # 相關路徑
    logs_path = './logs'

    # 載入模型
    print('載入模型中......\n')
    net = DnCNN(channels=1, num_of_layers=num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(logs_path, 'net.pth')))
    model.eval()

    # 載入資料
    print("載入測試資料中......\n")
    files = glob.glob(os.path.join('data', 'Set68', '*.png'))
    files.sort()

    # 測試階段
    psnr_test = 0
    for f in files:
        # 載入圖檔
        img = cv2.imread(f)
        img = np.float32(img[:, :, 0]) / 255.  # 將通道數設置為0
        img = np.expand_dims(img, 0)  # 增加batch_size，其值為1
        img = np.expand_dims(img, 1)  # 增加通道數，其值為1
        img_tensor = torch.Tensor(img)

        # 建立雜訊訊號
        noise = torch.FloatTensor(img_tensor.size()).normal_(mean=0, std=25./255.)
        # 含有雜訊的影像
        img_noise = img_tensor + noise
        img_tensor, img_noise = Variable(img_tensor).cuda(), Variable(img_noise).cuda()
        with torch.no_grad():
            restore_img = torch.clamp(img_noise-model(img_noise), 0., 1.)
        psnr = batch_psnr(restore_img, img_tensor, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files)
    print("\nPSNR on test data %f" % psnr_test)


if __name__ == '__main__':
    main()
