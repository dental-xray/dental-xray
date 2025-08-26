import kagglehub
import multiprocessing
import torch

# Download latest version
path = kagglehub.dataset_download("lokisilvres/dental-disease-panoramic-detection-dataset")

print("Path to dataset files:", path)

from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
# pt_file = path+"/best.pt"
pt_file = "yolo11n-seg.pt"
model = YOLO(pt_file)

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
yaml_file = path + "/YOLO/YOLO/data.yaml"
results = model.train(
    data=yaml_file,
    device='cuda',
    #epochs=1,
    #imgsz=416,      # 640 -> 416 に削減
    #batch=2,        # デフォルト16 -> 2 に削減
    #workers=1,      # ワーカー数削減
    #cache=False     # キャッシュ無効化
    #device='cpu',
    epochs=5,
    imgsz=416,                    # 小さめサイズ
    batch=8,                      # CPUなら大きなバッチも可能
    #workers=multiprocessing.cpu_count(),  # 全CPUコア活用
    #cache=True,                   # RAMキャッシュ
    #patience=50,
    #lr0=0.001,
    #optimizer='AdamW',
    #mosaic=0.5,
    #mixup=0.0,
    #degrees=5.0,
    #translate=0.05
)


# Run inference with the YOLOv8n model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")

save_dir = results.save_dir
print(f"Results saved in: {save_dir}")

best_model = results.save_dir / 'weights' / 'best.pt'
model.save('trained_model.pt')

#===== Loading a model =====
#loaded_model = YOLO('my_trained_model.pt')
#best_model = YOLO('runs/segment/train/weights/best.pt')
#results = loaded_model('test_image.jpg')

#===== Prediction =====
test_image = path + "/YOLO/YOLO/test/images/000dc27f-NAJIB_MARDANLOO_MASUME_2020-07-12185357_jpg.rf.adab6a526a5ec0225d2ce97c33095e8c.jpg"
test_results = model(test_image)
test_results.show()
#test_results.save('output/') 

