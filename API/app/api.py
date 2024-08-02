from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

origins = [
    "https://ui-service-dot-eloquent-env-430802-s8.uw.r.appspot.com"
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = YOLO("/Users/dimitridumont/code/skool/501/AAI-501/detect/train8/weights/best.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to our kickass application boi."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # Read the image file
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_np = np.array(image)

        # Run the image through the YOLO model
        results = model(image_np)

        # Process the model output to draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.numpy()  # Bounding boxes in (x1, y1, x2, y2) format
            scores = result.boxes.conf.numpy()  # Confidence scores
            labels = result.boxes.cls.numpy()  # Class labels

            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5:  # Only consider detections with a confidence score above 0.5
                    has_detections = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_np, f"{int(label)}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # if not has_detections:
        # # Draw "No Tumor" text on the image
        #     cv2.putText(image_np, "No Tumor", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Convert the image back to a byte stream
        is_success, buffer = cv2.imencode(".jpg", image_np)
        io_buf = io.BytesIO(buffer)

        return StreamingResponse(io_buf, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Processing error")