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
  pert_out = input + epsilon*data_grad.sign() 
  pert_out = torch.clamp(pert_out, 0, 1)
  return pert_out

# iterative fgsm attack
def ifgsm_attack(input,epsilon,data_grad):
  iter = 10
  # 한 번에 적용하는 공격의 크기
  alpha = epsilon/iter
  pert_out = input
  for i in range(iter-1):
    pert_out = pert_out + alpha*data_grad.sign()
    pert_out = torch.clamp(pert_out, 0, 1)
    # tensor.norm : 텐서의 크기를 계산
    # p = 1, 2, inf 등을 사용할 수 있음
    # p = 1 -> L1 norm
    # 절댓값 합 // 예시 : [3, -4] -> 3 + 4 = 7
    # p = 2 -> L2 norm
    # 유클리드 거리 // 예시 : [3, -4] -> sqrt(3^2 + (-4)^2) = 5
    # p = inf -> L-infinity norm
    # 최대 절댓값 // 예시 : [3, -4] -> max(|3|, |-4|) = 4

    #pert와 input의 차이가 epsilon보다 크면 중단
    if torch.norm((pert_out-input),p=float('inf')) > epsilon:
      break
  return pert_out