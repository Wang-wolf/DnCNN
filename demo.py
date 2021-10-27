import os
import cv2
import glob
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from models import DnCNN


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
    # 載入PNG影像
    print('載入影像中......')
    files = glob.glob(os.path.join('data', 'Demo', '*.png'))
    files.sort()

    # 開始處理影像
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (312, 416))
        img = np.float32(img[:, :, 0]) / 255.  # 將通道數設置為0
        img = np.expand_dims(img, 0)  # 增加batch_size，其值為1
        img = np.expand_dims(img, 1)  # 增加通道數，其值為1

        img_tensor = torch.Tensor(img)

        # 建立雜訊訊號
        noise = torch.FloatTensor(img_tensor.size()).normal_(mean=0, std=25. / 255.)
        # 含有雜訊的影像
        img_noise = img_tensor + noise
        img_tensor, img_noise = Variable(img_tensor).cuda(), Variable(img_noise).cuda()
        img_tensor, img_noise = img_tensor.cuda(), img_noise.cuda()
        with torch.no_grad():
            restore_img = torch.clamp(img_noise - model(img_noise), 0., 1.)
        img_tensor_show = img_tensor.squeeze(1).squeeze(0).cpu().numpy()
        img_noise_show = img_noise.squeeze(1).squeeze(0).cpu().numpy()
        restore_img_show = restore_img.squeeze(1).squeeze(0).cpu().numpy()

        plt.imshow(img_tensor_show, 'gray')
        plt.imshow(img_noise_show, 'gray')
        plt.imshow(restore_img_show, 'gray')
        plt.show()


if __name__ == '__main__':
    main()