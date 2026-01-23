# CIFAR-10 분류 모델들

import torch.nn as nn

# 첫번째 모델: Fully Connected Neural Network
class FullyConnectNN(nn.Module):
    def __init__(self):
        # 부모 클래스의 생성자를 호출 -> 부모가 가진 기능들 쓸 수 있도록!
        super(FullyConnectNN, self).__init__()
        # 레이어 정의하기
        # 32x32 크기의 컬러 이미지(3채널)를 1차원 벡터(3072)로 바꿈
        # 은닉층의 뉴런 개수는 512개로 설정!
        # 첫번째 은닉층
        self.fc1 = nn.Linear(32*32*3, 512) # 입력층 -> 은닉층
        # 두번째 은닉층
        self.fc2 = nn.Linear(512, 512) # 은닉층 -> 은닉층
        self.fc3 = nn.Linear(512, 10)  # 은닉층 -> 출력층 (10개 클래스 분류)
        # 활성화 함수로 ReLU 사용
        self.relu = nn.ReLU()

    # forward 메서드: 데이터가 모델을 통과할 때 어떻게 변하는지 정의
    def forward(self, x):
        x1 = self.fc1(x) # 첫번째 레이어 통과
        x2 = self.relu(x1) # ReLU 활성화 함수 적용
        x3 = self.fc2(x2) # 두번째 레이어 통과
        x4 = self.relu(x3) # ReLU 활성화 함수 적용
        x5 = self.fc3(x4) # 세번째 레이어 통과
        return x5


# 두번째 모델: Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째 합성곱층: 입력 채널 3개(컬러), 출력 채널 32개, 커널 크기 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 두번째 합성곱층: 입력 채널 32개, 출력 채널 64개, 커널 크기 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 세번째 합성곱층: 입력 채널 64개, 출력 채널 128개, 커널 크기 3x3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 풀링층: 2x2 크기의 맥스 풀링
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 완전 연결층: 입력 뉴런 128*4*4, 출력 뉴런 10 (10개 클래스 분류)
        self.fc1 = nn.Linear(128 * 4 * 4, 10)
        # 활성화 함수로 ReLU 사용
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x) # 첫번째 합성곱층 통과
        x = self.relu(x) # ReLU 활성화 함수 적용
        x = self.pool(x) # 풀링층 통과

        x = self.conv2(x) # 두번째 합성곱층 통과
        x = self.relu(x) # ReLU 활성화 함수 적용
        x = self.pool(x) # 풀링층 통과

        x = self.conv3(x) # 세번째 합성곱층 통과
        x = self.relu(x) # ReLU 활성화 함수 적용
        x = self.pool(x) # 풀링층 통과

        x = x.view(-1, 128 * 4 * 4) # fully connected layer에 넣기 위해 1차원 벡터로 쫙쫙 피기
        x = self.fc1(x) # 완전 연결층 통과
        return x
