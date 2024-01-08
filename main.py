from fastapi import FastAPI, WebSocket
from camera import Camera
import cv2
import base64
import uvicorn

from fastapi import FastAPI, WebSocket, HTTPException, status
from fastapi.responses import HTMLResponse
import pathlib
import datetime
import os

app = FastAPI()
camera = Camera()
save_dir = "data/1"
os.makedirs(save_dir, exist_ok=True)

import torch


def load_embeddings_from_file(file_path):
    return torch.load(file_path).numpy()


@app.get("/", response_class=HTMLResponse)
def read_root():
    html_file_path = pathlib.Path(__file__).parent / "index.html"
    with open(html_file_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        frame = camera.get_frame()
        if frame is not None:
            # Convert frame to base64 for web compatibility
            encoded_frame = base64.b64encode(frame).decode("utf-8")
            await websocket.send_text(encoded_frame)


@app.post("/save_image")
def save_image():
    frame, face_coords = camera.get_last_frame_and_face_coords()
    if frame is None or face_coords is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No face detected"
        )

    for i, (x0, y0, x1, y1) in enumerate(face_coords):
        cropped_face = frame[int(y0) : int(y1), int(x0) : int(x1)]
        filename = (
            f"data/1/face_{i}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(filename, cropped_face)

    return {"message": f"{len(face_coords)} face(s) saved"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
