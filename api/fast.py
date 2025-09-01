from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from api.main import run_prediction

app = FastAPI()

# Enable CORS (optional, useful for browser/JS requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"greeting": "Hello Teeth"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image and get YOLO prediction results.
    """
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Run prediction on the uploaded file
    detections = run_prediction(temp_path)

    # Print results in server console
    print(f"Prediction Results for {file.filename}: {detections}")

    # Return results as JSON
    return {
        "filename": file.filename,
        "detections": detections,
    }
