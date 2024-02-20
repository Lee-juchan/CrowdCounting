# An unoffical demo for training MCNN on Shanghai tech dataset (Part-B)
referece : https://www.kaggle.com/code/wptouxx/shtech-mcnn


# directory
폴더 구조  
```  
MCNN_fed/
    |____ ShanghaiTech/                 # ShanghaiTech dataset
    |           |__ part_A_final  
    |           |__ part_B_final  
    |  
    |____ datasets/  
    |           |__ data_sample.py      # data 샘플 확인  
    |           |__ dataset.py          # dataset, dataloader 생성, 테스트  
    |  
    |____ model.py                      # MCNN 모델  
    |____ train.py                      # MCNN 모델 훈련  
    |____ test.py                       # MCNN 모델 테스트 (단일 이미지)  
    |  
    |____ mcnn_model.pth                # 훈련된 MCNN 가중치  
    |  
    |____ client.py                     # MCNN client  
    |____ server.py                     # MCNN server  


    # other에는 원본 .ipynb 파일 존재
```

# requirements
필요 패키지 설치
```
pip install -r requirements.txt  
```

# run
- 단일 모델 train / test
```
python3 train.py
```  
```
python3 test.py
```  

- federated learning (서버-클라이언트)
```
python3 server
```
```
python3 client.py   # 각각 다른 쉘에 client 2개 이상 실행
```
