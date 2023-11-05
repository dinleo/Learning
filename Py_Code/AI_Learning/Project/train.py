# -*- coding: utf-8 -*-
"""
# train
#   이 파일을 실행시키면 checkpoint.pth 를 생성하므로, 실행시키기 전에 checkpoint.pth 를 지워야 함
#   모델 훈련(이 파일) 은 로컬에서 돌리기를 권장
# test
#   모델 테스트(test.ipynb) 는 colab 에서 돌리기를 권장
#   colab 에서 돌리려면 [파일] -> [노트 업로드] -> [test.ipynb] 노트가 열리면 우측 상단에 [연결]
#   런타임에 연결되면 좌측에 폴더모양 클릭후 " model.py, utils.py, val,npz, checkpoint.pth " 이 4개를 반드시 업로드 해줘야 함
#   그 후 실행
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import MyModel
from utils import score


# 학습에 필요한 상수들 설정
train_fp = 'train.npz'  # 학습 데이터 경로
test_fp = 'val.npz'
device = 'cpu'  # 사용할 장치
batch_size = 64  # 한번에 학습할 데이터 양
max_epoch = 25  # 전체 데이터를 총 20 번 학습

""" 
# 데이터 전처리 Class 인 CustomDataset 정의
# 인스턴스 생성시 파일경로를 받음
# for 문 돌리면 train_loader = [(이미지1, 라벨1), (이미지2, 라벨2), (이미지3, 라벨3), ... ] 형태로 접근할 수 있게됨
# 따라서 for 문 하나당 sample = (이미지1, 라벨1) 로 사용할 수 있게됨
"""
class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with np.load(file_path, allow_pickle=True) as data:
            self.data = data["data"]
            self.labels = data["labels"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_data = self.data[idx].astype("uint8").reshape((28, 28))
        img_label = int(self.labels[idx])

        img_data = Image.fromarray(img_data)

        if self.transform:
            img_data = self.transform(img_data)

        return img_data, img_label


# 이미지 변환처리를 해주는 트랜스포머 설정 자세한건 X
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.286], std=[0.353]),
    ]
)


""" 
# 위에서 정의한 "CustomDataset" 클래스(ex: Person)로 "train_dataset" 라는 인스턴스(ex: Mike) 생성
# 인스턴스 생성시 파일경로인 data_fp 와 위에서 설정한 transform 를 넣어줌
"""
train_dataset = CustomDataset(train_fp, transform=train_transforms)

"""
# torch 에서 제공하는 "DataLoader" 클래스로 "train_loader" 인스턴스를 생성
# "train_loader" 인스턴스 생성할때 위에서 생성한 "train_dataset" 를 넣어줌 
# ( train_loader 와 train_dataset 다름 주의 )
"""
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# test data 에도 동일하게 적용
test_dataset = CustomDataset(test_fp, transform=train_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


"""
모델 파라미터 count
"""
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if count_parameters(MyModel()) > 1000000:
    raise ValueError("Model parameter number should not exceed one million!")


"""
# 훈련 함수 정의
# model : 우리가 생성한 모델
# optimizer: 최적화함수 (역전파(=학습)을 시작하는 함수라고 생각하면 됨)
# sample: (데이터1, 라벨1)
"""
def train(model, optimizer, sample):
    # 오차 함수 생성
    criterion = nn.CrossEntropyLoss()
    # 학습모드로 전환
    model.train()

    # (데이터1, 라벨1) 을 각각 input, label 로 받음
    input, label = sample[0].to(device), sample[1].to(device)

    # 모델 순전파(=예측) 진행
    # 예측결과 pred = [ 0.1, 0.4, 0.3, ... ] 확률 벡터함수 반환
    pred = model(input)

    # 오차함수에 pred 와 정답인 label을 넣어 오차 계산
    loss = criterion(pred, label)

    # 맞은 갯수 계산
    num_correct = torch.sum(torch.argmax(pred, dim=-1) == label)

    # optimizer 초기화
    optimizer.zero_grad()
    # 역전파 진행
    loss.backward()
    # 모델 parameter 업데이트 (= 학습)
    optimizer.step()

    return loss.item(), num_correct.item()


def test(model, sample):
    model.eval()

    with torch.no_grad():
        input, label = sample[0].to(device), sample[1].to(device)
        pred = model(input)
        num_correct = torch.sum(torch.argmax(pred, dim=-1) == label)

    return num_correct.item()
"""
# optimizer 설정
# 여기서는 가장 보편적으로 많이 쓰는 Adam 사용
"""
def get_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)


"""
# 모델 구현은 model.py 에 되어있음
# 따라서 from model import MyModel 을 통해 model.py 에 있는 MyModel 클래스를 불러옴
# 원래 여기있던 MyModel 코드는 중복이므로 필요 없어서 지움
"""

# 불러온 MyModel 로 모델 생성
model = MyModel().to(device)
# 학습을 위한 optimizer 생성
optimizer = get_optimizer(model)
# 모델 구조 출력
print(model)
print(f"모델 파라미터 갯수: {count_parameters(MyModel())}")
# 훈련 시작
for epoch in range(max_epoch):
    avg_tr_loss = 0.0
    avg_tr_correct = 0.0

    # train_loader = [(이미지1, 라벨1), (이미지2, 라벨2), (이미지3, 라벨3), ... ]
    # sample = (이미지1, 라벨1)
    for sample in train_loader:
        # 위에서 설정한 train 함수로 모델 학습 batch 수만큼(=64개) 의 sample 로 진행
        tr_loss, tr_correct = train(model, optimizer, sample)

        avg_tr_loss += tr_loss / len(train_loader)
        avg_tr_correct += tr_correct / len(train_dataset)

    # 총 데이터가 657 개이고, 64개씩 진행하므로 약 11번 반복후, 1 epoch 끝남
    # max_epoch=10 이므로 이걸 10 번 반복

    print('[EPOCH {}] TRAINING LOSS : {:.02f}, TRAINING ACCU : {:.02f}'.format(epoch + 1, avg_tr_loss,
                                                                               avg_tr_correct * 100))
    avg_te_correct = 0
    for sample in test_loader:
        te_correct = test(model, sample)
        avg_te_correct += te_correct / len(test_dataset)

    print('TEST ACCU: {:.02f}%'.format(avg_te_correct * 100))
    print('TEST SCORE: {:.02f} out of 100'.format(score(avg_te_correct * 100)))
    if (epoch+1) % 5 == 0 and epoch !=0:
        checkpoint = {
            'model': MyModel(),
            'model_state_dict': model.state_dict(),
        }
        torch.save(checkpoint, f'checkpoint{epoch+1}.pth')

# 모델 클래스 정보인 MyModel 과 학습된 모델 파라미터인 model.state_dict() 를 저장
# checkpoint = {
#     'model': MyModel(),
#     'model_state_dict': model.state_dict(),
# }
# torch.save(checkpoint, 'checkpoint.pth')
