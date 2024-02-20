'''
    Prepare dataset : Dataset 정의, Dataloader 생성
'''

import os
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A


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
        
        '''여기 밑은 손을 못보겠다.'''
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


# train, valid, test 나누기 (img path들만 들어있음)
train = [p.path for p in os.scandir('./ShanghaiTech/part_B/train_data/images/')]     # list comprehension 사용한 버전
test_full = [p.path for p in os.scandir('./ShanghaiTech/part_B/test_data/images/')]

test, valid = train_test_split(test_full, test_size=64, random_state=42) # test_full -> test, valid


# argument
img_size = 512
aug_train = A.Compose([
    A.RandomCrop(img_size, img_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(),
    A.Normalize((0.5), (0.5)),
], keypoint_params=A.KeypointParams(format='xy', angle_in_degrees=False))

aug_val = A.Compose([
    A.Resize(768, 1024),
    A.Normalize((0.5), (0.5)),
], keypoint_params=A.KeypointParams(format='xy', angle_in_degrees=False))


# dataloader 생성
def load_data(batch_size):
    train_loader = DataLoader(MyDataset(train, aug_train), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2) # pin_memory= : CUDA 고정 메모리에 올릴지여부
    val_loader = DataLoader(MyDataset(valid, aug_val), batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)   # num_workers= : 데이터 로딩에 사용하는 subprocess 수 (멀티 프로세싱), 원본은 4
    test_loader = DataLoader(MyDataset(test, aug_val), batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)

    return train_loader, val_loader, test_loader



# dataset, dataloader 테스트
if __name__ == "__main__":
    # data path list
    print(len(train), len(valid), len(test)) # 400 64 252

    # dataset
    dataset = MyDataset(train, aug_train) # train sample
    img, gt = dataset[0]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 5)
    ax[0].set_title('image')
    ax[0].imshow(img, cmap='gray')
    ax[1].set_title('ground truth')
    ax[1].imshow(gt)
    plt.show()

    # dataloader
    _, val_loader, _ = load_data(batch_size=32)

    for idx, (imgs, labels) in enumerate(val_loader):
        print(f"{idx}/{len(val_loader)}", end=' ')
        print("x shape:", imgs.shape, end=' ')      # (32, 768, 1024) 맨 앞은 batch_size
        print("y shape:", labels.shape)