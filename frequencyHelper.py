
import numpy as np
from scipy import signal
import torch 


def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def torch_fftshift(img):
    fft = torch.fft.fft2(img)
    return torch.fft.fftshift(fft)

def torch_ifftshift(fft):
    fft = torch.fft.ifftshift(fft)
    return torch.fft.ifft2(fft)


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result




masks=None
def generateDataWithDifferentFrequencies_3Channel(Images,radius,device):
    global masks
    if masks is None:
        masks={}
        for r in range(1, 256 ,1):
            rows, cols = Images.size(2) , Images.size(3)
            mask = torch.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    dis = np.sqrt((i - rows/2) ** 2 + (j - rows/2) ** 2)
                    if dis < r:
                        mask[i, j] = 1.0
                
            masks[r] = mask

    mask = masks[radius]
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)
    fd = torch_fftshift(Images)
    
    low_freq_img = fd*mask
    low_freq_img = torch_ifftshift(low_freq_img)
    
    high_freq_img = fd*(1-mask)
    high_freq_img = torch_ifftshift(high_freq_img)

    return low_freq_img, high_freq_img

if __name__ == '__main__':
    x = torch.randn(size=(1,3,32,32))
    y =x.permute(0,2,3,1)
    c , d = generateDataWithDifferentFrequencies_3Channel(x,14,torch.device("cpu"))
   
    
    print(torch.sum((c)).numpy().astype(int))
    #print(masks[4])