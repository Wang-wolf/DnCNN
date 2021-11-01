# 影像處理-影像降躁化(去躁化)
(Image Processing - Make Noise Images Clean)  
得力於電腦效能的大幅提升以及GPU的平行運算架構，讓我們能夠更快速且有效地訓練AI，並將AI技術應用於不同領域。本篇將帶給大家的是 **「將深度學習應用於影像處理中的影像降躁化
」**，透過影像降躁的處理能得較為「平順」、「平滑」的影像，這些經過處理的影像可以進行後續的分析或是深度學習的模型訓練。  
* 文獻參考 : *[Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189)*  
* 實作參考 : *[DnCNN-pytorch](https://github.com/SaoYan/DnCNN-PyTorch)*  
* 論文作者實作 : *[MATLAB implementation](https://github.com/cszn/DnCNN)*


## API安裝(Build environment)
此專案所使用得API版本  
(API version for this project)
* Pytorch(torch): 1.9.0+cu102
* Numpy : 1.21.2
* Skimage : 0.16.2
* TensorboradX : 2.4
### Notice:
* Skimage的版本不是最新，若使用最新版本將無法使用`compare_psnr`函式.  
( Skimage version is not the latest. You can't call the function`compare_psnr` if you use the latest one. )
* 可以透過`pip3 install`來安裝所需要的API.  
(You can install APIs with the instruction`pip3 install`)
