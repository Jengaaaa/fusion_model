import torch
import torch.nn as nn
import timm
from dataset import load_video_frames, load_mfcc_chunks



class SimpleVideoEncoder(nn.Module):
    def __init__(self, model_name="inception_resnet_v2", out_dim=256):
        super().__init__()

        # timm에서 백본 생성 , timm은 pretrained 모델 불러오게 해주는 라이브러리
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=True
        )
        self.backbone.classifier = nn.Identity() 
        
        # 분류층(fc) 제거, 목표: 우울증 회귀 모델이기 때문에
        #backbone은 핵심 feature 표현 뽑아내는 네트워크 전체 ex) resnet, 특징추출기

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Inception-ResNet-V2 feature dim = 1536
        self.fc = nn.Linear(1536, out_dim)
        #256차원 feature vector로 바꾸기.

        

    def forward(self, frames):
        """
        frames: (B, T, 3, 299, 299)  프레임 형
        """
        B, T, C, H, W = frames.shape

        # (B*T, 3, 299, 299)
        #프레임 레벨에서 feature 추출
        #모든 프레임을 일괄 CNN에 넣기
        flat_frames = frames.view(B * T, C, H, W)

        # CNN feature 추출
        feat = self.backbone.forward_features(flat_frames)
        feat = self.pool(feat).view(B * T, -1)   # (B*T, 1536) 
        feat = self.fc(feat)                    # (B*T, 256)

        # (B, T, 256)으로 reshape
        feat = feat.view(B, T, -1)

        # 모든 프레임 평균 → (B, 256)
        video_embedding = feat.mean(dim=1) #평균 풀링

        return video_embedding
