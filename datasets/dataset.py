'''
    Dataset 정의
'''

import numpy as np
import scipy
import cv2
import torch
from torch.utils.data import Dataset


# dataset
class ShanghaiTech(Dataset):
    def __init__(self, files, aug):
        self.files = files      # file name list
        self.aug = aug          # transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fn = self.files[idx] # file name
       
        # file load (IMG, GT)
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mat = scipy.io.loadmat(fn.replace('images', 'ground-truth').replace('IMG', 'GT_IMG').replace('.jpg', '.mat'))
        pos = mat['image_info'][0][0][0][0][0] # mat file -> (x,y) 좌표
        
        # transform 적용
        auged = self.aug(image=img, keypoints=pos)
        img = auged['image']
        pos = auged['keypoints']
        
        # density map 생성 (좌표들 -> img)
        dm = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        for x, y in pos:
            x = int(x)
            y = int(y)
            dm[y, x] = 1    # dm[row, column]

        # ground truth downsampling
        sigma = 4
        dm = cv2.GaussianBlur(dm, (0, 0), sigma)
        dm = cv2.resize(dm, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_LINEAR) # interpolation= : 보간법
        dm *= 16
        
        img = torch.from_numpy(img)
        dm = torch.from_numpy(dm)
        
        return img, dm


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
        A.Normalize(mean=(0.5), std=(0.5)),
    ], keypoint_params=A.KeypointParams(format='xy', angle_in_degrees=False))

    # dataset 생성
    dataset = ShanghaiTech(train, aug_train) # train sample
    img, dm = dataset[0]
    # print(img.shape, dm.shape)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 5)
    ax[0].set_title('image')
    ax[0].imshow(img, cmap='gray')
    ax[1].set_title('ground truth')
    ax[1].imshow(dm)
    plt.show()