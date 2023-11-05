import torch.nn as nn

""" 

# Convolution: 행렬합성곱 연산을 수행하는 레이어 이미지 분류에 뛰어난 성능을 보임
    (1, 28, 28) 이었던 이미지가 (out_channels=32, kernel_size=3) 인 Convolution Layer 를 거치면
    (32, 26, 26) 이 됨. 이미지 shape 계산 공식은 google 에 검색하면 나옴
# MaxPooling: 이미지 가로세로 사이즈를 kernel_size(=2 배) 만큼 줄여서 계산량 감소
    (32, 24, 24) -> (32, 12, 12)
# ReLU: 활성화 함수
# Flatten: 다음 Layer 가 선형이므로 넘기기전에 이미지를 쭉 펴줘야함

"""

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.cv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        """ x 를 레이어 순서대로 순전파 연속으로 진행"""
        x = self.cv_layer1(x)
        x = self.cv_layer2(x)
        x = self.fc_layer(x)

        return x