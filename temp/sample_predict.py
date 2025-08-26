import kagglehub
import multiprocessing
import torch

# Download latest version
path = kagglehub.dataset_download("lokisilvres/dental-disease-panoramic-detection-dataset")

print("Path to dataset files:", path)

from ultralytics import YOLO

#===== Loading a model =====
pt_file = "trained_model.pt"
model = YOLO(pt_file)
model.info()

#===== Prediction =====
test_image = path + "/YOLO/YOLO/test/images/000dc27f-NAJIB_MARDANLOO_MASUME_2020-07-12185357_jpg.rf.adab6a526a5ec0225d2ce97c33095e8c.jpg"
results = model.predict(test_image)
print(results)
#test_results.save('output/') 

