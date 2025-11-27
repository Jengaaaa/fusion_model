import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)

    def forward(self, x):
        identity = x  # skip connection
        
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        return self.relu(out + identity)



class SimpleAudioEncoder(nn.Module):
    def __init__(self, n_mfcc=40, out_dim=256):   #40차원의 음성 특
        super().__init__()

        # 1D CNN layers, MFCC시간축을 따라 움직임
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # 최종 FC layer (256 → 256)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, mfcc):
        """
        mfcc shape:
            (40, T) 또는 (B, 40, T)
        """

        # 배치가 없는 경우 → 배치 축 추가
        if mfcc.dim() == 2:
            mfcc = mfcc.unsqueeze(0)   # (1, 40, T)

        # CNN feature extraction
        x = self.cnn(mfcc)             # (B, 256, T)

        # Global Average Pooling (T축 평균) /GAP, 하나의 시간축을 평균내서 고정길이 벡터로
        x = x.mean(dim=2)              # (B, 256)

        # 최종 256D 벡터
        x = self.fc(x)                 # (B, 256)
        return x
