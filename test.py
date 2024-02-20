'''
    Visualize the result    : 학습된 모델의 결과 확인
'''
''' output 이미지를 통해 사람 수를 계산해야함'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from model import MCNN


# model 가중치 불러오기
lr = 3e-4
model = MCNN(lr=lr)
model.load_state_dict(torch.load('./mcnn_model.pth'))


# 결과 test
# input image
im_path = '../ShanghaiTech/part_B/test_data/images/IMG_1.jpg'
im = cv2.imread(im_path, cv2.IMREAD_COLOR)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# output image
im = im.astype(np.float32) / 255.0              # [0,1]로 정규화
im_tensor = torch.from_numpy(im).unsqueeze(0)   # batch_size 차원 추가 : (w, h) -> (1, w, h)

out = model(im_tensor)
out_im = out.detach().squeeze(0)                # batch_size 차원 제거 : (1, w, h) -> (w, h)

# plot
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(10,5)
ax[0].set_title('input image')
ax[0].imshow(im, cmap='gray')

ax[1].set_title('output image')
ax[1].imshow(out_im)
plt.show()

# save output image
plt.savefig('model_output.png')