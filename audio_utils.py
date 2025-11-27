# audio_utils.py
# 적용을 위한 모듈 분할

import numpy as np
import torch
import librosa

def load_mfcc_chunks(path, n_mfcc=40, chunk_len=3000, hop_ratio=1.0):
    """
    path: wav 파일 경로
    chunk_len: one chunk covers ~30s of MFCC (depending on hop size)
    hop_ratio: 1.0 
    """

    y, sr = librosa.load(path, sr=16000)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)  # (40, T)

    T = mfcc.shape[1]
    hop = int(chunk_len * hop_ratio)

    chunks = []
    for start in range(0, T, hop):
        end = start + chunk_len
        piece = mfcc[:, start:end]

        if piece.shape[1] < chunk_len:
            pad = chunk_len - piece.shape[1]
            piece = np.pad(piece, ((0, 0), (0, pad)), mode="constant")

        chunks.append(torch.tensor(piece, dtype=torch.float32))

        if end >= T:
            break

    return chunks
