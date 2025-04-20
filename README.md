# 📷 ESP-CAM Depth API

A FastAPI server that receives images from an ESP32-CAM and estimates depth or water level based on image processing.

## 🚀 Features

- Accepts JPEG image uploads via HTTP POST.
- Uses OpenCV to process images.
- Provides JSON response with estimated water level (placeholder logic).

---

## 🧰 Requirements

- Python 3.8+
- pip

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/esp-cam-depth-api.git
cd esp-cam-depth-api
```
2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 📡 Run the server

```bash
uvicorn main:app --reload --port 8000
```

The server will be running at `http://localhost:8000/upload`.

## 📤 Example Usage (CLI)
You can use `curl` to send a test image to the server:

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@path/to/image.jpg"
  ```
