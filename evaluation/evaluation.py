import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import cv2
import argparse

from natsort import natsort
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from niqe import niqe
from piqe import piqe
from brisque import BRISQUE
os.environ['CUDA_VISIBLE_DEVICES']='1'
class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)
        self.brisque_obj = BRISQUE(url=False)
    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips, self.EN_A,self.EN_B,self.AG_A, self.AG_B, self.niqe_B, self.piqe_B, self.brisque]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        score, diff = ssim(imgA, imgB, full=True, multichannel=True,channel_axis=2)
        return score

    def psnr(self, imgA, imgB):
        psnr_val = psnr(imgA, imgB)
        return psnr_val
    
    def EN_A(self,imgA, imgB):
        #caculate histgram  
        imgA = np.array(imgA).astype(np.int32)
        histogram, bins = np.histogram(imgA, bins=256, range=(0, 255))
        # hist normalization
        histogram = histogram / float(np.sum(histogram))
        # caculate entrpoy_A
        entrpoy_A = -np.sum(histogram * np.log2(histogram + 1e-7))
        return entrpoy_A
    
    def EN_B(self,imgA, imgB):
        imgB = np.array(imgB).astype(np.int32)
        histogram, bins = np.histogram(imgB, bins=256, range=(0, 255))
        # hist normalization
        histogram = histogram / float(np.sum(histogram))
        # caculate entrpoy_A
        entropy_B = -np.sum(histogram * np.log2(histogram + 1e-7))
        return entropy_B
    
    def AG_A(self,imgA, imgB):
        imgA = np.array(imgA).astype(np.float32)
        # print(imgA.shape)
        width = imgA.shape[1]
        width = width - 1
        height = imgA.shape[0]
        height = height - 1
        AG_A = 0
        for i in (0,2):
            grad_x = np.gradient(imgA[:, :, i])  # 在X轴方向上计算通道0的梯度
            grad_y = np.gradient(imgA[:, :, i])  # 在Y轴方向上计算通道0的梯度`
            avg_grad = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))
            AG_A = avg_grad + AG_A
        AG_A = np.sum(AG_A)/3
        return AG_A
    
    def AG_B(self,imgA, imgB):
        imgB = np.array(imgB).astype(np.float32)
        # print(imgA.shape)
        width = imgB.shape[1]
        width = width - 1
        height = imgB.shape[0]
        height = height - 1
        AG_B = 0
        for i in (0,2):
            grad_x = np.gradient(imgB[:, :, i])  # 在X轴方向上计算通道0的梯度
            grad_y = np.gradient(imgB[:, :, i])  # 在Y轴方向上计算通道0的梯度`
            avg_grad = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))
            AG_B = avg_grad + AG_B
        AG_B = np.sum(AG_B)/3
        return AG_B
    
    def niqe_B(self, imgA,imgB):
        # np.array(Image.open(path_gt).convert('LA'))[:,:,0]
        niqe_score = 0
        for i in (0,2):   
            # print("shape",imgB.shape)         
            imgB_rgb = ((imgB[:,:,i]) * 255.0).round().astype(np.uint8)
            imgB_rgb = np.squeeze(imgB_rgb)
            # print(imgB.shape)  #256,256,3
            niqe_score += niqe(imgB_rgb)
        return niqe_score
    
    def piqe_B(self, imgA,imgB):
        piqe_score = piqe(imgB)
        return piqe_score

    def brisque(self, imgA, imgB):
        brisque_score = self.brisque_obj.score(imgB)
        return brisque_score
def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips, EN_A,EN_B, AG_A,AG_B,niqe_B,piqe_B,brisque_B):
    return f'{psnr:0.4f}, {ssim:0.4f}, {lpips:0.4f},{EN_A:0.4f},{EN_B:0.4f},{AG_A:0.4f},{AG_B:0.4f},{niqe_B:0.4f},{piqe_B:0.4f},{brisque_B:0.4f}'

def measure_dirs(dirA, dirB, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)

    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()
        img_A = imread(pathA)
        img_B = imread(pathB)
        weight_B, height_B, channel_B=img_B.shape
        img_A = cv2.resize(img_A, (weight_B, height_B), cv2.INTER_AREA)
        t = time.time()
        result['psnr'], result['ssim'], result['lpips'], result['EN_A'] ,result['EN_B'] ,result['AG_A'],result['AG_B'],result['niqe_B'],result['piqe_B'], result['brisque_B']= measure.measure(img_A, img_B)
        d = time.time() - t
        # vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")

        results.append(result)

    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])
    EN_A = np.mean([result['EN_A'] for result in results])
    EN_var_A = np.var([result['EN_A'] for result in results])
    AG_A = np.mean([result['AG_A'] for result in results])
    AG_var_A = np.var([result['AG_A'] for result in results])
    EN_B = np.mean([result['EN_B'] for result in results])
    EN_var_B = np.var([result['EN_B'] for result in results])
    AG_B = np.mean([result['AG_B'] for result in results])
    AG_var_B = np.var([result['AG_B'] for result in results])
    niqe_B = np.mean([result['niqe_B'] for result in results])
    piqe_B = np.mean([result['piqe_B'] for result in results])
    brisque_B = np.mean([result['brisque_B'] for result in results])
    print("EN_mean_A:", EN_A,"EN_mean_B:",EN_B )
    print("EN_var_A:", EN_var_A,"EN_var_B:",EN_var_B )
    print("AG_mean_A:", AG_A, "AG_mean_B:",AG_B )
    print("AG_var_A:", AG_var_A, "AG_var_B:",AG_var_B )
    print("---------------------------NONE Reference Score----------------------------")
    print("lpips:",lpips,"Niqe_B:",niqe_B,"Piqe_B:",piqe_B)
    print("----------------------------------------------------------------------------")
    vprint(f"Final Result: {format_result(psnr, ssim, lpips, EN_A, AG_A,EN_B, AG_B,niqe_B,piqe_B,brisque_B)}, {time.time() - t_init:0.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('-dirA', default='/data/tanqiaozhi/LLCaps/LA-Net-main/snapshots/kc/external-enhanced-image', type=str)
    #parser.add_argument('-dirA', default='/data/tanqiaozhi/LLCaps/Data/KC/eval15/high', type=str)
    #parser.add_argument('-dirA', default='/data/tanqiaozhi/LLCaps/Data/RLE/evaluation/high', type=str)
    parser.add_argument('-dirA', default='/data/tanqiaozhi/LLCaps/Data/EC-0129/test/origin', type=str)
    # parser.add_argument('-dirA', default='/mnt/data-hdd2/bailong/Low-light-Enhancement/SNR-Aware-Low-Light-Enhance/results/LOLv1_RLE/images/GT', type=str)
    #CFWD-KC
    parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/PYDIFF/experiments/infer_EC_origin_28.6/visualization', type=str)
    # parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/LOL/CFWD-main/output/KC-enhanced/Setdataset256/5000', type=str)  #35.776
    # parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/LOL/CFWD-main/output/KC-enhanced/Setdataset256/6000', type=str) # 35.7490
    # parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/LOL/Diffusion-Low-Light-main/results/LLdataset256/5000', type=str)
    # parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/EC/PromptIR-main/output/RLE/image-enhanced-2/derain', type=str)
    # parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/EC/LACT-main/log/result_0131/enhanced-image', type=str)
    # parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/CLEDiffusion-main/output/EC-result/result/epoch1000', type=str)
    # parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/LA-Net-main/snapshots/kc/external-enhanced-image', type=str)
    # parser.add_argument('-dirB', default='/data/tanqiaozhi/LLCaps/LOL/CFWD-main/output/RLE-enhanced/Setdataset256/5000', type=str)


    parser.add_argument('-type', default='png')
    parser.add_argument('--use_gpu', default=True)
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    type = args.type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, use_gpu=use_gpu, verbose=True)
