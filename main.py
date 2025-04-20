from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io

app = FastAPI()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Call water level / depth estimation function
    level = estimate_water_level(img)

    return {"estimated_depth": level}

def estimate_water_level(image):
    # Dummy logic for now
    height, width = image.shape[:2]
    print(f"Received image of size: {width}x{height}")

    # Placeholder: return a fake depth value
    return 42.0
