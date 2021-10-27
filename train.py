import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as tvutils

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import prepare_data, Dataset
from models import DnCNN, weights_init_kaiming
from utils import batch_psnr


def main():
    # 參數設置
    num_workers = 2
    batch_size = 128
    num_of_layers = 17  # 模型總層數
    lr = 1e-3
    epochs = 20
    milestone = 10

    # 路徑設定
    logs_path = './logs'

    print("正在載入資料集......")
    train_dataset = Dataset(train=True)
    val_dataset = Dataset(train=False)

    # 使用Pytorch中的載入器(DataLoader)，建立訓練集載入器
    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers,
                              batch_size=batch_size, shuffle=True)
    print("訓練樣本數:%d\n" % len(train_dataset))

    # 建立DnCNN模型
    net = DnCNN(channels=1, num_of_layers=num_of_layers)
    net.apply(weights_init_kaiming)
    # 建立Loss Function
    criterion = nn.MSELoss(size_average=False)

    # 移至GPU運算
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()

    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 訓練模型
    writer = SummaryWriter(logs_path)
    step = 0
    noise_level_blind = [0, 55]
    for epoch in range(epochs):
        if epoch < milestone:
            current_lr = lr
        else:
            current_lr = lr / 10.

        # 設置學習率
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("學習率:%f" % current_lr)

        for i, data in enumerate(train_loader, 0):
            model.train()
            # 每個批次梯度是累加的
            # 開始每個批次訓練之前需要歸零
            model.zero_grad()
            optimizer.zero_grad()

            img_train = data
            noise = torch.zeros(img_train.size())
            # 在0~55之間產生連續型均勻分布
            std_n = np.random.uniform(noise_level_blind[0], noise_level_blind[1],
                                      size=noise.size()[0])
            for n in range(noise.size()[0]):
                size_n = noise[0, :, :, :].size()
                # 生成一個大小維度與size_n相同的張量
                # 每次用不同的標準差(std_n[n])進行正規化
                noise[n, :, :, :] = torch.FloatTensor(size_n).normal_(mean=0, std=std_n[n]/255.)
            # 加入雜訊至原始影像
            img_train_noise = img_train + noise
            img_train, img_train_noise = Variable(img_train.cuda()), Variable(img_train_noise.cuda())
            noise = Variable(noise.cuda())
            # Residual Learning
            # 不讓神經網路直接學習原始影像與加入雜訊影像之間的關聯
            # 而是學習真實雜訊
            output_train = model(img_train_noise)
            loss = criterion(output_train, noise) / (img_train_noise.size()[0]*2)
            loss.backward()
            optimizer.step()

            # 批次訓練結果
            model.eval()
            # 將輸出箝制在0.0~1.0
            restore_img = torch.clamp(img_train_noise-model(img_train_noise), 0., 1.)
            psnr_train = batch_psnr(restore_img, img_train, 1.)
            print("[Epoch %d][%d/%d] Loss:%.4f, PSNR_train:%.4f" %
                  (epoch+1, i+1, len(train_loader), loss.item(), psnr_train))

            # 每十個批次紀錄
            if step % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        # 每個epoch結束後，驗證一下
        # 驗證階段

        model.eval()
        psnr_val = 0
        # 單張驗證
        for k in range(len(val_dataset)):
            img_val = torch.unsqueeze(val_dataset[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=25./255.)
            img_val_noise = img_val + noise
            img_val, img_val_noise = Variable(img_val.cuda(), volatile=True), Variable(img_val_noise.cuda(), volatile=True)
            restore_val = torch.clamp(img_val_noise-model(img_val_noise), 0., 1.)
            psnr_val += batch_psnr(restore_val, img_val, 1.)
        psnr_val = psnr_val / len(val_dataset)
        print("\n[epoch %d] PSNR_val:%.4f" % (epoch+1, psnr_val))

        # 紀錄每個epoch的模型表現
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        output_train = torch.clamp(img_train_noise-model(img_train_noise), 0., 1.)
        image = tvutils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        image_noise = tvutils.make_grid(img_train_noise, nrow=8, normalize=True, scale_each=True)
        restore_image = tvutils.make_grid(output_train.data, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Clean image', image, epoch)
        writer.add_image('Noisy image', image_noise, epoch)
        writer.add_image('Restore image', restore_image, epoch)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(logs_path, 'net.pth'))


if __name__ == '__main__':
    # prepare_data(data_path='./data', patch_size=50, stride=10, aug_times=2)
    main()
