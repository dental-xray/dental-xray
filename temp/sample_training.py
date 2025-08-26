import kagglehub
from ultralytics import YOLO
import os

# あなたの既存のトレーニングコード
path = kagglehub.dataset_download("lokisilvres/dental-disease-panoramic-detection-dataset")
pt_file = path + "/yolo11n-seg.pt"
model = YOLO(pt_file)

yaml_file = path + "/YOLO/YOLO/data.yaml"
results = model.train(
    data=yaml_file, 
    device='cuda', 
    epochs=1, 
    imgsz=640,
    project='dental_detection',    # プロジェクト名
    name='experiment_1',          # 実験名
    save=True
)

print(f"トレーニング完了！結果は {results.save_dir} に保存されました")

# === 保存されたファイルの場所 ===
best_model_path = results.save_dir / 'weights' / 'best.pt'
last_model_path = results.save_dir / 'weights' / 'last.pt'

print(f"ベストモデル: {best_model_path}")
print(f"最新モデル: {last_model_path}")

# === モデルのロードと使用 ===
# 推論用（最高性能モデル）
inference_model = YOLO(str(best_model_path))

# トレーニング継続用（最新チェックポイント）  
resume_model = YOLO(str(last_model_path))

# 推論テスト
test_image = "path/to/your/test/image.jpg"
if os.path.exists(test_image):
    results = inference_model(test_image)
    results.show()  # 結果表示
    results.save('inference_results/')  # 結果保存

# トレーニング継続
# continue_results = resume_model.train(
#     data=yaml_file,
#     epochs=5,  # 追加で5エポック
#     resume=True
# )
