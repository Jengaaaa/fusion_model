import os
import uuid
import torch
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from fusion import FusionNet
from simple_video import load_video_frames
from audio_utils import load_mfcc_chunks


app = FastAPI(title="Depression Score API", version="1.0")


# --------------------------------------------------------
# Load Model
# --------------------------------------------------------
MODEL_PATH = "checkpoints/best_fusionnet.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FusionNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✔ Model loaded successfully!")


# --------------------------------------------------------
# Utility: mp4 → wav
# --------------------------------------------------------
def extract_wav(mp4_path):
    wav_path = mp4_path.replace(".mp4", ".wav")
    cmd = f'ffmpeg -y -i "{mp4_path}" -vn -ac 1 -ar 16000 "{wav_path}"'
    subprocess.call(cmd, shell=True)
    return wav_path


# --------------------------------------------------------
# Score interpretation
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
# Prediction function
# --------------------------------------------------------
def run_inference(mp4_path):
    # 1) WAV extract
    wav_path = extract_wav(mp4_path)

    # 2) Load video frames
    frames = load_video_frames(mp4_path, num_frames=8)
    frames = frames.unsqueeze(0).to(device)

    # 3) Load MFCC Chunks
    mfcc_chunks = load_mfcc_chunks(wav_path)

    # 4) Inference
    preds = []
    with torch.no_grad():
        for chunk in mfcc_chunks:
            chunk = chunk.unsqueeze(0).to(device)
            out = model(frames, chunk)
            preds.append(out.item())

    score = sum(preds) / len(preds)
    level = interpret_score(score)

    # Temporary wav 삭제
    if os.path.exists(wav_path):
        os.remove(wav_path)

    return score, level


# --------------------------------------------------------
# API: /predict
# --------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) mp4 저장 경로
    upload_id = str(uuid.uuid4())
    save_path = f"temp_{upload_id}.mp4"

    # 2) 파일 저장
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # 3) 추론
    score, level = run_inference(save_path)

    # 4) temp mp4 삭제
    if os.path.exists(save_path):
        os.remove(save_path)

    # 5) 결과 반환
    return JSONResponse({
        "depression_score": score,
        "interpretation": level
    })
