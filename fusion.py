import torch
import torch.nn as nn
from simple_video import SimpleVideoEncoder
from simple_audio import SimpleAudioEncoder


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Encoders ---
        self.video_encoder = SimpleVideoEncoder()   # (B, 256)
        self.audio_encoder = SimpleAudioEncoder()   # (B, 256)

        # --- Attention Gate ---
        #가중치 생성 게이트 -> 비디오, 오디오 중에 더 중요한 것 자동 판단
        # x = concat(video, audio) = (B, 512)
        # attention weight scalar (per sample)
        self.attention_gate = nn.Sequential(
            nn.Linear(512, 256),         #두 모달리티 정보 섞어 컴팩트한 특징
            nn.ReLU(),                   #비선형성 추가 (학습 성능 향상)
            nn.Linear(256, 1),
            nn.Sigmoid()               # weight ∈ (0, 1)
        )

        # --- Fusion Regressor (Deep MLP) ---
        #우울 점수(연속값) 예측 MLP
        #단순 변경된점, noise 감소, modality간의 조합 정교한 학습.
        self.regressor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)        #최종 우울 점수 1개 출력
        )

    def forward(self, frames, mfcc):
        """
        frames: (B, T, 3, 299, 299)  #비디오 프레임 T개
        mfcc:   (B, 40, T)           #mfcc 시간축 길이 t`
        """
        # --- Encode ---
        v = self.video_encoder(frames)  # (B, 256)
        a = self.audio_encoder(mfcc)    # (B, 256)  #반환된 embedding들

        # --- Concatenate ---
        x = torch.cat([v, a], dim=1)    # (B, 512)  #결합

        # --- Attention Fusion (video/audio 가중치 조절) ---
        w = self.attention_gate(x)      # (B, 1)

        fused = w * v + (1 - w) * a     # (B, 256) #어텐션이 결정한 비율만큼 섞기 
        #음성,영상 조절 해서 반영

        # 해석:
        # w ≈ 1 → video 중심
        # w ≈ 0 → audio 중심
        # w ≈ 0.5 → 둘 다 적절히 결합

        # --- Final Fusion Input ---
        final_input = torch.cat([fused, v, a], dim=1)   # (B, 256+256+256=768)
        #attention 결과 + 순수 비디오 정보 + 순수 오디오 정보

        # BUT regressor는 512 input으로 설계됨 768에서 512로 축소
        # 선택: fused(256) + x(512) 중 512만 사용
        final_input = x     # (B, 512) 

        # --- Regressor ---
        out = self.regressor(final_input)  # (B, 1) #우울 점수 예측

        return out.squeeze(1) #결정
