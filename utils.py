import numpy as np
from skimage.measure.simple_metrics import compare_psnr


def batch_psnr(image, imclean, data_range):
    img = image.data.cpu().numpy().astype(np.float32)
    img_clean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img.shape[0]):
        psnr += compare_psnr(img_clean[i, :, :, :], img[i, :, :, :], data_range=data_range)
    return psnr/img.shape[0]
