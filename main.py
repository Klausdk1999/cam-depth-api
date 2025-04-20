from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import torchvision.transforms as T

app = FastAPI()

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval()

# Define transform manually for DPT models (MiDaS v3)
transform = T.Compose([
    T.Resize(384),
    T.CenterCrop(384),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.get("/status")
def status():
    return {"status": "ok", "model": "MiDaS DPT_Large"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Read image with PIL
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_input = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return JSONResponse({
        "depth_min": float(depth_map.min()),
        "depth_max": float(depth_map.max()),
        "depth_mean": float(depth_map.mean()),
    })
