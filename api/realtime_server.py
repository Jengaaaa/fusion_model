# api/realtime_server.py

import os
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from datetime import datetime
import torch

# --------------------------------------------
# Import (프로젝트 root에서 실행된다는 전제)
# --------------------------------------------
from fusion import FusionNet
from audio_utils import wav_to_mfcc_realtime   # 반드시 audio_utils.py에 추가해야 함

app = FastAPI(title="Real-Time Depression Analysis API")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------
# 모델 로드
# --------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # / 프로젝트 root
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints/best_fusionnet.pth")

model = FusionNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("🔥 FusionNet loaded on", device)


# --------------------------------------------
# 세션 저장소
# --------------------------------------------
sessions = {}

def init_session(user_id):
    sessions[user_id] = {
        "frame_buffer": [],
        "audio_buffer": np.array([], dtype=np.float32),
        "latest_score": None,
        "latest_level": None,
        "updated_at": None
    }


# --------------------------------------------
# 점수 해석 함수
# --------------------------------------------
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


# --------------------------------------------
# FusionNet 실시간 추론 함수
# --------------------------------------------
def maybe_predict(user_id):
    session = sessions[user_id]

    # 조건 부족하면 return
    if len(session["frame_buffer"]) < 8:
        return
    if len(session["audio_buffer"]) < 16000:
        return

    # ------------------------------
    # 1) Frame 준비
    # ------------------------------
    frames = session["frame_buffer"][-8:]

    processed = []
    for img in frames:
        img = cv2.resize(img, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        processed.append(img)

    frames_tensor = torch.tensor(processed, dtype=torch.float32)
    frames_tensor = frames_tensor.unsqueeze(0).to(device)

    # ------------------------------
    # 2) Audio 준비: 1초 떼기
    # ------------------------------
    audio_1sec = session["audio_buffer"][:16000]
    session["audio_buffer"] = session["audio_buffer"][16000:]

    mfcc = wav_to_mfcc_realtime(audio_1sec)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

    # ------------------------------
    # 3) 모델 추론
    # ------------------------------
    with torch.no_grad():
        out = model(frames_tensor, mfcc_tensor)
        score = float(out.item())

    level = interpret_score(score)

    session["latest_score"] = round(score, 2)
    session["latest_level"] = level
    session["updated_at"] = datetime.utcnow().isoformat()

    print(f"[Predict] {score:.2f} ({level})")


# --------------------------------------------
# 1) /realtime/init
# --------------------------------------------
@app.post("/realtime/init")
def realtime_init(user_id: str):
    init_session(user_id)
    return {"status": "initialized", "user_id": user_id}


# --------------------------------------------
# 2) /realtime/frame
# --------------------------------------------
@app.post("/realtime/frame")
async def realtime_frame(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    if user_id not in sessions:
        return JSONResponse(status_code=400, content={"status": "error", "message": "세션 없음"})

    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "error", "message": "프레임 디코딩 실패"}

    session = sessions[user_id]
    session["frame_buffer"].append(frame)
    if len(session["frame_buffer"]) > 8:
        session["frame_buffer"].pop(0)

    maybe_predict(user_id)

    return {
        "status": "ok",
        "frame_buffer_len": len(session["frame_buffer"])
    }


# --------------------------------------------
# 3) /realtime/audio
# --------------------------------------------
@app.post("/realtime/audio")
async def realtime_audio(
    user_id: str = Form(...),
    audio_chunk: UploadFile = File(...)
):
    if user_id not in sessions:
        return JSONResponse(status_code=400, content={"status": "error", "message": "세션 없음"})

    audio_bytes = await audio_chunk.read()
    pcm = np.frombuffer(audio_bytes, dtype=np.float32)

    session = sessions[user_id]
    session["audio_buffer"] = np.concatenate([session["audio_buffer"], pcm])

    maybe_predict(user_id)

    sec = len(session["audio_buffer"]) / 16000.0
    return {"status": "ok", "audio_buffer_sec": round(sec, 2)}


# --------------------------------------------
# 4) /realtime/result
# --------------------------------------------
@app.get("/realtime/result")
def realtime_result(user_id: str):
    if user_id not in sessions:
        return JSONResponse(status_code=404, content={"status": "error", "message": "세션 없음"})

    s = sessions[user_id]
    return {
        "score": s["latest_score"],
        "level": s["latest_level"],
        "updated_at": s["updated_at"]
    }


# --------------------------------------------
# 5) /realtime/reset
# --------------------------------------------
@app.post("/realtime/reset")
def realtime_reset(user_id: str):
    if user_id not in sessions:
        return JSONResponse(status_code=404, content={"status": "error", "message": "세션 없음"})

    init_session(user_id)
    return {"status": "reset", "user_id": user_id}
