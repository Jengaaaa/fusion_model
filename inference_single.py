import os
import torch
import subprocess
from tqdm import tqdm

from simple_video import load_video_frames
from audio_utils import load_mfcc_chunks
from fusion import FusionNet


# --------------------------------------------------------
# mp4 → wav 변환
# --------------------------------------------------------
def extract_wav_from_mp4(mp4_path, wav_path):
    cmd = f'ffmpeg -y -i "{mp4_path}" -vn -ac 1 -ar 16000 "{wav_path}"'
    subprocess.call(cmd, shell=True)


# --------------------------------------------------------
# 우울 점수 해석
# --------------------------------------------------------
def interpret_score(s):
    if s < 10:
        return "정상 범위 (우울감 거의 없음)"
    elif s < 20:
        return "경미한 우울 수준 가능성"
    elif s < 30:
        return "중등도 우울 가능성"
    elif s < 40:
        return "높은 우울 수준"
    else:
        return "매우 높은 우울 수준"


# --------------------------------------------------------
# 단일 비디오 예측
# --------------------------------------------------------
def predict_from_video(mp4_path, model_path="checkpoints/best_fusionnet.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load model
    print("[INFO] Loading model...")
    model = FusionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✔ Model loaded")

    # Extract WAV
    wav_path = mp4_path.replace(".mp4", ".wav")
    print("[INFO] Extracting audio...")
    extract_wav_from_mp4(mp4_path, wav_path)

    # Load video
    print("[INFO] Loading video frames...")
    frames = load_video_frames(mp4_path, num_frames=8)
    frames = frames.unsqueeze(0).to(device)
    print("✔ Video loaded.")

    # Load audio MFCC
    print("[INFO] Loading MFCC chunks...")
    mfcc_chunks = load_mfcc_chunks(wav_path)
    print(f"✔ MFCC chunks: {len(mfcc_chunks)}")

    # Predict
    print("[INFO] Running prediction...")
    preds = []

    with torch.no_grad():
        for chunk in tqdm(mfcc_chunks, desc="Predicting"):
            chunk = chunk.unsqueeze(0).to(device)
            out = model(frames, chunk)
            preds.append(out.item())

    score = sum(preds) / len(preds)
    level = interpret_score(score)

    print("\n===================================")
    print(f" Predicted Depression Score: {score:.2f}")
    print(f" Interpretation: {level}")
    print("===================================\n")

    # Remove WAV
    if os.path.exists(wav_path):
        os.remove(wav_path)
        print("[INFO] Removed temp wav file.")

    return score, level


# --------------------------------------------------------
# 실행
# --------------------------------------------------------
if __name__ == "__main__":
    print("### Inference Script Started ###")

    video_path = input("Enter mp4 path: ").strip()
    video_path = video_path.replace("\ufeff", "").replace("ㅅ", "")

    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        exit()

    predict_from_video(video_path)
