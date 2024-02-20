'''
    visualize   : shaghaitech B 데이터셋 샘플 확인
'''

import matplotlib.pyplot as plt
import scipy

def show(img, gt):
    # positions of ground truth
    pos = gt['image_info'][0][0][0][0][0]   # .mat에서 좌표만 추출 
                                            # gt['image_info'][0][0][0][0][1] => 좌표(점)수
    X = pos[:, 0]
    Y = pos[:, 1]

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 5)
    ax[0].set_title('image')
    ax[0].imshow(img)

    ax[1].set_title('ground truth')
    ax[1].imshow(img)
    ax[1].scatter(X, Y, marker="+", c="r")
    plt.show()



# test
if __name__ == '__main__':
    img = plt.imread('./ShanghaiTech/part_B/train_data/images/IMG_1.jpg') # image
    gt = scipy.io.loadmat('./ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_1.mat') # ground truth
    show(img, gt)