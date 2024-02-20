'''
    Prepare dataset : Dataset 정의
'''

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import cv2
import torch
from torch.utils.data import Dataset


# dataset
class MyDataset(Dataset):
    def __init__(self, files, aug):
        self.files = files
        self.aug = aug
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fn = self.files[idx] # file path
        
        im = cv2.imread(fn, cv2.IMREAD_COLOR) # img (gray)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        
        m = scipy.io.loadmat(fn.replace('images', 'ground-truth').replace('IMG', 'GT_IMG').replace('.jpg', '.mat')) # ground truth
        ps = m['image_info'][0][0][0][0][0]
        
        rst = self.aug(image=im, keypoints=ps)
        im = rst['image']
        ps = rst['keypoints']
        
        dm = np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)
        for x, y in ps:
            x = int(x)
            y = int(y)
            dm[y, x] = 1

        sigma = 4
        dm = gaussian_filter(dm, sigma=sigma, truncate=4*sigma)
        dm = cv2.resize(dm, (im.shape[1] // 4, im.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
        dm *= 16
        
        im = torch.from_numpy(im)
        dm = torch.from_numpy(dm)
        
        return im, dm


# dataset test
if __name__ == '__main__':
    import os
    import os.path as path
    import albumentations as A
    import matplotlib.pyplot as plt

    # path list
    train = [p.path for p in os.scandir(path.join(path.dirname(__file__), '../ShanghaiTech/part_B/train_data/images/'))]

    # augment
    img_size = 512
    aug_train = A.Compose([
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(),
        A.Normalize((0.5), (0.5)),
    ], keypoint_params=A.KeypointParams(format='xy', angle_in_degrees=False))

    # dataset 생성
    dataset = MyDataset(train, aug_train) # train sample
    img, gt = dataset[0]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 5)
    ax[0].set_title('image')
    ax[0].imshow(img, cmap='gray')
    ax[1].set_title('ground truth')
    ax[1].imshow(gt)
    plt.show()