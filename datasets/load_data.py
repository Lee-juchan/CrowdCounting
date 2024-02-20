'''
    Prepare dataset : Dataloader 생성
'''

import os
import os.path as path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A

from dataset import MyDataset


# data set 분리 (img path만 저장된)
train = [p.path for p in os.scandir(path.join(path.dirname(__file__), '../ShanghaiTech/part_B/train_data/images/'))]    # list comprehension 사용한 버전
test_full = [p.path for p in os.scandir(path.join(path.dirname(__file__), '../ShanghaiTech/part_B/test_data/images/'))]
test, valid = train_test_split(test_full, test_size=64, random_state=42) # test_full -> test + valid

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
'''train, val, test 생성 함수를 분리할까??'''
batch_size = 32
train_loader = DataLoader(MyDataset(train, aug_train), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=2) # pin_memory= : CUDA 고정 메모리에 올릴지여부
val_loader = DataLoader(MyDataset(valid, aug_val), batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)   # num_workers= : 데이터 로딩에 사용하는 subprocess 수 (멀티 프로세싱), 원본은 4
test_loader = DataLoader(MyDataset(test, aug_val), batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)

def load_data():
    return train_loader, val_loader, test_loader


# dataloader 테스트
if __name__ == "__main__":
    # img path list
    print(len(train), len(valid), len(test)) # 400 64 252

    # dataloader
    for idx, (imgs, gts) in enumerate(val_loader):
        print(f"{idx}/{len(val_loader)}", end=' ')
        print("x shape:", imgs.shape, end=' ')      # (32, 768, 1024) 맨 앞은 batch_size
        print("y shape:", gts.shape)