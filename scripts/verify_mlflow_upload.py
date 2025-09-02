import mlflow.artifacts
import tempfile
import os
from disease_recognition.params import *
from mlflow import MlflowClient


def verify_mlflow_upload(mlflow_model_name):
    """Verify if the model is correctly uploaded to MLflow and can be downloaded"""

    print("=== MLflow Upload Verification ===")
    print(f"mlflow_model_name: {mlflow_model_name}")

    try:
        client = MlflowClient()

        latest_versions = client.get_latest_versions(mlflow_model_name, stages=["None"])

        if not latest_versions:
            print("❌ No model found in registry")
            return False

        model_version = latest_versions[0]
        print(f"✅ Model found: {mlflow_model_name}")
        print(f"  Version: {model_version.version}")
        print(f"  Status: {model_version.status}")
        print(f"  Source: {model_version.source}")

        print("\n🔄 Testing download capability...")
        return test_model_download_enhanced(mlflow_model_name, model_version.version)

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def test_model_download_enhanced(mlflow_model_name, version):
    """Enhanced test to download and inspect the model artifacts from MLflow"""

    print(f"=== Testing download for model: {mlflow_model_name}, version: {version} ===")
    print(f"mlflow_model_name: {mlflow_model_name}")
    print(f"version: {version}")

    try:
        model_uri = f"models:/{mlflow_model_name}/{version}"
        print(f"Downloading from: {model_uri}")

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=temp_dir)
            print(f"Downloaded to: {artifact_path}")

            print("\n=== Downloaded Structure ===")
            all_files = []
            pt_files = []

            for root, dirs, files in os.walk(artifact_path):
                level = root.replace(artifact_path, '').count(os.sep)
                indent = '  ' * level
                print(f"{indent}{os.path.basename(root)}/")

                subindent = '  ' * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    rel_path = os.path.relpath(file_path, artifact_path)

                    print(f"{subindent}{file} ({file_size:,} bytes)")
                    all_files.append(rel_path)

                    if file.endswith('.pt'):
                        pt_files.append(file_path)
                        print(f"{subindent}🎯 PyTorch model found!")

            print(f"\n=== Summary ===")
            print(f"Total files: {len(all_files)}")
            print(f"All files: {all_files}")

            if pt_files:
                print(f"✅ Found {len(pt_files)} .pt file(s):")
                for pt_file in pt_files:
                    size = os.path.getsize(pt_file)
                    print(f"  - {os.path.basename(pt_file)}: {size:,} bytes")

                    try:
                        from ultralytics import YOLO
                        test_model = YOLO(pt_file)
                        print(f"  ✅ YOLO loading test successful")
                        return True
                    except Exception as yolo_error:
                        print(f"  ⚠️  YOLO loading failed: {yolo_error}")

                return True
            else:
                print("❌ No .pt files found anywhere")

                mlmodel_path = os.path.join(artifact_path, "MLmodel")
                if os.path.exists(mlmodel_path):
                    print("\n=== MLmodel Content ===")
                    with open(mlmodel_path, 'r') as f:
                        content = f.read()
                        print(content)

                return False

    except Exception as e:
        print(f"❌ Download test failed: {e}")
        return False

def comprehensive_mlflow_check(mlflow_model_name):
    """Perform a comprehensive check of the MLflow model registry"""

    print("=== Comprehensive MLflow Check ===")
    print(f"mlflow_model_name: {mlflow_model_name}")

    client = MlflowClient()

    try:
        registered_model = client.get_registered_model(mlflow_model_name)
        print(f"✅ Registered model: {registered_model.name}")
        print(f"  Description: {registered_model.description}")
        print(f"  Tags: {registered_model.tags}")

    except Exception as e:
        print(f"⚠️  Cannot get registered model info: {e}")

    try:
        all_versions = client.search_model_versions(f"name='{mlflow_model_name}'")
        print(f"\n📋 All versions ({len(all_versions)}):")

        for i, version in enumerate(all_versions):
            print(f"  Version {version.version}:")
            print(f"    Status: {version.status}")
            print(f"    Stage: {version.current_stage}")
            print(f"    Source: {version.source}")
            print(f"    Run ID: {getattr(version, 'run_id', 'N/A')}")

            if i == 0:
                print(f"    Testing version {version.version}...")
                test_result = test_model_download_enhanced(mlflow_model_name, version.version)
                print(f"    Download test: {'✅ PASS' if test_result else '❌ FAIL'}")

    except Exception as e:
        print(f"⚠️  Cannot list versions: {e}")

    return True

def main():
    """Main function with enhanced verification"""

    print(f"Target model: {MLFLOW_MODEL_NAME}")

    basic_result = verify_mlflow_upload(MLFLOW_MODEL_NAME)
    comprehensive_mlflow_check(MLFLOW_MODEL_NAME)

if __name__ == "__main__":
    main()
