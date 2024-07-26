from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3002"
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to our kickass application boi."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        # Read the image file
        image = Image.open(file.file)
        
        # Example processing: Convert image to grayscale
        grayscale_image = image.convert('L')
        
        # Save processed image to a BytesIO object
        img_io = io.BytesIO()
        grayscale_image.save(img_io, 'JPEG')
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Processing error")