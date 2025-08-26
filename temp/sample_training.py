import kagglehub

# Download latest version
path = kagglehub.dataset_download("lokisilvres/dental-disease-panoramic-detection-dataset")

print("Path to dataset files:", path)

from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
# pt_file = path+"/best.pt"
pt_file = path+"/yolo11n-seg.pt"
model = YOLO(pt_file)

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
yaml_file = path + "/YOLO/YOLO/data.yaml"
results = model.train(data=yaml_file, epochs=1, imgsz=640)
