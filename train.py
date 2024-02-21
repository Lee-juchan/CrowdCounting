'''
    Training
'''

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.dataloader import load_data
from model import MCNN

seed_everything(42)

# data loader 준비
batch_size = 32
train_loader, val_loader, test_loader = load_data(batch_size=batch_size)

# trainer 생성
epochs = 10
max_steps = epochs * len(train_loader) # len(train_loader) == len(train) // batch_size

lr_monitor = LearningRateMonitor()
checkpoint_cb = ModelCheckpoint(
    save_top_k=1,
    save_last=True,
    verbose=True,
    monitor='val_mae',
    mode='min',
)

trainer = Trainer(max_steps=max_steps, precision=16, benchmark=True, callbacks=[checkpoint_cb, lr_monitor]) # gpus=1 없으면 default = cpu / benchmark= : input 크기가 동일하면 시스템 속도 증가

# training
'''weight init is crucial for the model to converge'''
def weights_normal_init(model, std=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, std) # 재귀적으로 모든 param 초기화
    else:
        for module_name, module in model.named_modules(): # named_modules() : 자신은 포함한 모든 sub module 반환
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)


lr = 3e-4
model = MCNN(lr, max_steps=max_steps)

weights_normal_init(model, std=0.01)
model.use = 0

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# added
# Here's the code to save your model
torch.save(model.state_dict(), 'mcnn_model_1.pth')


'''Separate column training'''
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# seed_everything(42)

# checkpoint_cb = ModelCheckpoint(
#     save_top_k=1,
#     save_last=True,
#     verbose=True,
#     monitor='val_mae',
#     mode='min',
#     prefix=''
# )

# trainer = Trainer(gpus=1, max_steps=max_steps, precision=32, callbacks=[checkpoint_cb, LearningRateMonitor()])

# lr = 3e-4

# model = MCNN(lr, batch_size, max_steps)
# weights_normal_init(model, dev=0.01)

# model.use = 1

# trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

'''same x 3'''
# checkpoint_cb = ModelCheckpoint(
#     save_top_k=1,
#     save_last=True,
#     verbose=True,
#     monitor='val_mae',
#     mode='min',
#     prefix=''
# )

# trainer = Trainer(gpus=1, max_steps=max_steps, precision=32, callbacks=[checkpoint_cb, LearningRateMonitor()])

# model.use = 2 # 3, 0

# trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)