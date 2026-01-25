import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets

def fgsm_attack(input,epsilon,data_grad):
  # gradient에 대해 sign 함수 적용
  # 공격의 크기를 epsilon으로 조절
  # 한 번에 공격 적용
  pert_out = input + epsilon*(data_grad.sign()) 
  pert_out = torch.clamp(pert_out, 0, 1)
  return pert_out


def ifgsm_attack(model, input, label, epsilon, iters=10):
  # PGD 공격 - 매 iteration마다 gradient를 새로 계산
  
  alpha = epsilon / iters
  loss_function = nn.CrossEntropyLoss()
  
  # 초기 perturbation (원본 이미지에서 시작)
  pert_out = input.clone().detach()
  # clone : tensor 복사본 생성
  # detach : 그래디언트 계산에서 분리

  #매 iteration마다 새로운 gradient 계산
  for i in range(iters):
    # gradient 계산을 위해 requires_grad 설정
    pert_out.requires_grad = True
    
    # 모델 출력 계산
    output = model(pert_out)
    
    # 배치 차원 처리
    if label.dim() == 0:
      label_use = label.unsqueeze(0)
    else:
      label_use = label
    
    # loss 계산
    loss = loss_function(output, label_use)
    
    # gradient 계산
    model.zero_grad()

    # 매 iteration마다 gradient 새로 나와야함
    # 그러니 기존 gradient 초기화
    if pert_out.grad is not None:
      pert_out.grad.zero_()
    loss.backward()
    
    # gradient 방향으로 한 스텝 이동
    with torch.no_grad():
      pert_out = pert_out + alpha * pert_out.grad.sign()
      
      # epsilon ball로 projection (원본 이미지 기준)
      eta = torch.clamp(pert_out - input, min=-epsilon, max=epsilon)
      pert_out = torch.clamp(input + eta, min=0, max=1)

  return pert_out























def pgd_attack(model, input, label, epsilon, iters=10):
  # PGD 공격 - 매 iteration마다 gradient를 새로 계산
  
  alpha = epsilon / iters
  loss_function = nn.CrossEntropyLoss()
  
  # 초기 perturbation (랜덤 노이즈 추가 - PGD의 핵심!)
  # I-FGSM과의 차이점: 랜덤하게 시작
  pert_out = input.clone().detach()
  # uniform_(-epsilon, epsilon): -epsilon ~ +epsilon 사이의 랜덤 값으로 초기화
  pert_out = pert_out + torch.empty_like(input).uniform_(-epsilon, epsilon)
  pert_out = torch.clamp(pert_out, 0, 1)  # 픽셀 범위 유지

  #매 iteration마다 새로운 gradient 계산
  for i in range(iters):
    # gradient 계산을 위해 requires_grad 설정
    pert_out.requires_grad = True
    
    # 모델 출력 계산
    output = model(pert_out)
    
    # 배치 차원 처리
    if label.dim() == 0:
      label_use = label.unsqueeze(0)
    else:
      label_use = label
    
    # loss 계산
    loss = loss_function(output, label_use)
    
    # gradient 계산
    model.zero_grad()

    # 매 iteration마다 gradient 새로 나와야함
    # 그러니 기존 gradient 초기화
    if pert_out.grad is not None:
      pert_out.grad.zero_()
    loss.backward()
    
    # gradient 방향으로 한 스텝 이동
    with torch.no_grad():
      pert_out = pert_out + alpha * pert_out.grad.sign()
      
      # epsilon ball로 projection (원본 이미지 기준)
      eta = torch.clamp(pert_out - input, min=-epsilon, max=epsilon)
      pert_out = torch.clamp(input + eta, min=0, max=1)

  return pert_out