### 라이브러리 임포트 ###
import os
import cv2
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------
# 1) 영상(mp4) → 프레임 텐서 (T, 3, 299, 299)
# ---------------------------------------------------------
def load_video_frames(path, num_frames=8):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수

    # 영상을 num_frames(=32)개 지점으로 균등 분할 → 인덱스 배열 생성
    idx = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

    frames = []

    # 각 선택된 프레임 위치로 이동하여 프레임 읽기
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()

        # 프레임을 못 읽으면 (예: 영상 깨짐) → 검은 화면으로 대체
        if not ret:
            f = np.zeros((299, 299, 3), np.uint8)

        # OpenCV는 BGR, 우리는 RGB 사용 → 색상 채널 변환
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        # Inception-ResNet 입력 크기에 맞추어 리사이즈
        f = cv2.resize(f, (299, 299))
        frames.append(f)

    cap.release()

    # 파이썬 리스트 → numpy 배열 (T, H, W, 3)
    frames = np.stack(frames)
    # numpy → tensor, (T, H, W, C) → (T, C, H, W)로 순서 변경
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.
    return frames


# ---------------------------------------------------------
# 2) 오디오(wav) → MFCC를 sliding window로 여러 chunk로 자르기
#    반환: [ (40, chunk_len), (40, chunk_len), ... ]
# ---------------------------------------------------------
def load_mfcc_chunks(path, n_mfcc=40, chunk_len=3000, hop_ratio=1.0):
    """
    chunk_len : 3000이면 약 30초 MFCC 길이 (hop_length에 따라 다르지만 대략)
    hop_ratio : 1.0 = 겹침 없음 (0~3000, 3000~6000, ...)
    """

    # wav 파일 로드 (샘플링: 16000Hz)
    y, sr = librosa.load(path, sr=16000)

    # mel spectrogram 생성
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    # dB scale로 변환
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # MFCC 생성 (40, T)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)

    T = mfcc.shape[1]
    hop = int(chunk_len * hop_ratio)

    chunks = []

    # 0, hop, 2*hop, ... 위치에서 chunk_len 길이씩 잘라냄
    for start in range(0, T, hop):
        end = start + chunk_len
        piece = mfcc[:, start:end]

        # 마지막 chunk가 짧으면 zero-padding
        if piece.shape[1] < chunk_len:
            pad_width = chunk_len - piece.shape[1]
            piece = np.pad(piece, ((0, 0), (0, pad_width)), mode="constant")

        chunks.append(torch.tensor(piece, dtype=torch.float32))

        if end >= T:
            break

    return chunks   # 리스트: [ (40, chunk_len), ... ]


# ---------------------------------------------------------
# 3) AVEC Dataset (Training / Testing 공용)
# ---------------------------------------------------------
class AVECDataset(Dataset):
    def __init__(self, base_path, mode="Training", tasks=["Freeform", "Northwind"]):
        """
        base_path : /home/.../AVEC2014_AudioVisual
        mode      : Training or Testing 변경 가능
        tasks     : ["Freeform", "Northwind"] 둘 다 사용 가능
        """

        self.base_path = base_path
        self.mode = mode
        self.tasks = tasks  # 리스트 형태

        # (Training일 때만 사용)
        self.label_dir = os.path.join(
            base_path.replace("AVEC2014_AudioVisual", "Final_Labels"),
            f"{mode}_DepressionLabels"
        )

        # prefix 목록 생성
        self.samples = []   # [(task, prefix), ...]

        for task in tasks:
            video_dir = os.path.join(base_path, "Video", mode, task)
            for f in os.listdir(video_dir):
                if f.endswith(f"_{task}_video.mp4"):
                    prefix = f.split(f"_{task}_")[0]  # 예: 203_1
                    self.samples.append((task, prefix))

        # 정렬 (203_1, 204_2, …)
        self.samples = sorted(self.samples, key=lambda x: x[1])

        self.is_test = (mode == "Testing")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        task, prefix = self.samples[idx]

        # 파일 경로
        video_dir = os.path.join(self.base_path, "Video", self.mode, task)
        audio_dir = os.path.join(self.base_path, "Audio", self.mode, task)

        vpath = os.path.join(video_dir, f"{prefix}_{task}_video.mp4")
        apath = os.path.join(audio_dir, f"{prefix}_{task}_audio.wav")

        # 1) 비디오 로드
        frames = load_video_frames(vpath)

        # 2) 슬라이딩 MFCC chunk 생성
        mfcc_chunks = load_mfcc_chunks(apath, chunk_len=3000)

        # ---------- Testing 모드 ----------
        if self.is_test:
            # 여러 chunk 그대로 반환
            return frames, mfcc_chunks, f"{task}_{prefix}"

        # ---------- Training 모드 ----------
        # 라벨 로드
        label_path = os.path.join(self.label_dir, f"{prefix}_Depression.csv")
        label = float(open(label_path).read().strip())

        # 하나의 chunk만 사용 (첫 chunk), 가장 안정적인 정보 이용 -> 성능 안나오면 개선 예정
        mfcc_first = mfcc_chunks[0]

        return frames, mfcc_first, torch.tensor(label, dtype=torch.float32)

