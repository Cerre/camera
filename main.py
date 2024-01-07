from fastapi import FastAPI, WebSocket
from camera import Camera
import cv2
import base64

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pathlib

app = FastAPI()
camera = Camera()

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
            encoded_frame = base64.b64encode(frame).decode('utf-8')
            await websocket.send_text(encoded_frame)

