import mlflow
from mlflow import MlflowClient
import mlflow.artifacts
import tempfile
import os
import yaml
import json
from datetime import datetime

def analyze_mlflow_model_structure(mlflow_model_name, mlflow_tracking_uri, version=None):
    """MLflowモデルの詳細構造分析"""

    print("="*60)
    print("MLflow Model Structure Analysis")
    print("="*60)

    # MLflow接続
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    try:
        # 1. モデルレジストリ情報の確認
        print("\n### 1. Model Registry Information ###")
        try:
            registered_model = client.get_registered_model(mlflow_model_name)
            print(f"Model Name: {registered_model.name}")
            print(f"Description: {registered_model.description}")
            print(f"Tags: {registered_model.tags}")
        except Exception as e:
            print(f"❌ Cannot get registered model info: {e}")

        # 2. 全バージョンの確認
        print(f"\n### 2. All Model Versions ###")
        try:
            all_versions = client.search_model_versions(f"name='{mlflow_model_name}'")
            print(f"Total versions: {len(all_versions)}")

            for version_info in sorted(all_versions, key=lambda x: int(x.version)):
                print(f"\n  Version {version_info.version}:")
                print(f"    Status: {version_info.status}")
                print(f"    Stage: {version_info.current_stage}")
                print(f"    Source: {version_info.source}")
                print(f"    Run ID: {getattr(version_info, 'run_id', 'N/A')}")
                print(f"    Creation Time: {version_info.creation_timestamp}")

        except Exception as e:
            print(f"❌ Cannot list versions: {e}")
            return False

        # 3. 指定バージョンの詳細分析（または最新）
        if version is None:
            latest_versions = client.get_latest_versions(mlflow_model_name, stages=["None"])
            if latest_versions:
                version = latest_versions[0].version
            else:
                print("❌ No versions found")
                return False

        print(f"\n### 3. Detailed Analysis of Version {version} ###")
        model_uri = f"models:/{mlflow_model_name}/{version}"
        print(f"Model URI: {model_uri}")

        # 4. アーティファクト構造の分析
        return analyze_model_artifacts(model_uri, client, mlflow_model_name, version)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def analyze_model_artifacts(model_uri, client, mlflow_model_name, version):
    """アーティファクト詳細分析"""

    print(f"\n### 4. Artifact Structure Analysis ###")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading artifacts to: {temp_dir}")

            # アーティファクトダウンロード
            try:
                artifact_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=temp_dir)
                print(f"✅ Download successful: {artifact_path}")
            except Exception as download_error:
                print(f"❌ Download failed: {download_error}")
                return False

            # ディレクトリ構造の表示
            print(f"\n### 5. Directory Structure ###")
            display_directory_tree(artifact_path)

            # MLmodelファイルの分析
            print(f"\n### 6. MLmodel File Analysis ###")
            analyze_mlmodel_file(artifact_path)

            # ファイル詳細情報
            print(f"\n### 7. File Details ###")
            analyze_file_details(artifact_path)

            # モデルフレーバーの確認
            print(f"\n### 8. Model Flavor Detection ###")
            detect_model_flavors(artifact_path)

            # .ptファイルの検索
            print(f"\n### 9. PyTorch (.pt) File Search ###")
            find_pytorch_files(artifact_path)

            return True

    except Exception as e:
        print(f"❌ Artifact analysis failed: {e}")
        return False

def display_directory_tree(root_path, prefix="", max_depth=5, current_depth=0):
    """ディレクトリツリー表示"""
    if current_depth >= max_depth:
        return

    try:
        items = sorted(os.listdir(root_path))
        for i, item in enumerate(items):
            item_path = os.path.join(root_path, item)
            is_last = i == len(items) - 1

            if os.path.isdir(item_path):
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                extension = "    " if is_last else "│   "
                display_directory_tree(item_path, prefix + extension, max_depth, current_depth + 1)
            else:
                file_size = os.path.getsize(item_path)
                print(f"{prefix}{'└── ' if is_last else '├── '}{item} ({format_file_size(file_size)})")

    except Exception as e:
        print(f"{prefix}❌ Error reading directory: {e}")

def format_file_size(size_bytes):
    """ファイルサイズの読みやすい形式に変換"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def analyze_mlmodel_file(artifact_path):
    """MLmodelファイルの分析"""
    mlmodel_path = os.path.join(artifact_path, "MLmodel")

    if os.path.exists(mlmodel_path):
        try:
            with open(mlmodel_path, 'r') as f:
                mlmodel_content = yaml.safe_load(f)

            print("✅ MLmodel file found")
            print(f"MLflow Version: {mlmodel_content.get('mlflow_version', 'Unknown')}")
            print(f"Model UUID: {mlmodel_content.get('model_uuid', 'Unknown')}")
            print(f"Run ID: {mlmodel_content.get('run_id', 'Unknown')}")

            # フレーバー情報
            flavors = mlmodel_content.get('flavors', {})
            print(f"Available Flavors: {list(flavors.keys())}")

            for flavor_name, flavor_info in flavors.items():
                print(f"\n  {flavor_name.upper()} Flavor:")
                for key, value in flavor_info.items():
                    print(f"    {key}: {value}")

        except Exception as e:
            print(f"❌ Error reading MLmodel file: {e}")
    else:
        print("❌ MLmodel file not found")

def analyze_file_details(artifact_path):
    """各ファイルの詳細分析"""
    important_files = []

    for root, dirs, files in os.walk(artifact_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, artifact_path)
            file_size = os.path.getsize(file_path)

            # 重要なファイルタイプ
            if any(file.endswith(ext) for ext in ['.pt', '.pkl', '.yaml', '.yml', '.txt', '.json']):
                important_files.append({
                    'path': rel_path,
                    'full_path': file_path,
                    'size': file_size,
                    'ext': os.path.splitext(file)[1]
                })

    # サイズ順でソート
    important_files.sort(key=lambda x: x['size'], reverse=True)

    for file_info in important_files:
        print(f"  {file_info['path']} ({format_file_size(file_info['size'])})")

        # 特定のファイル内容を確認
        if file_info['ext'] in ['.yaml', '.yml', '.txt'] and file_info['size'] < 10000:
            try:
                with open(file_info['full_path'], 'r') as f:
                    content = f.read()[:500]  # 最初の500文字
                    print(f"    Content preview: {content[:200]}...")
            except:
                pass

def detect_model_flavors(artifact_path):
    """モデルフレーバーの検出"""
    flavors_found = []

    # PyTorch検出
    pytorch_files = []
    for root, dirs, files in os.walk(artifact_path):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                pytorch_files.append(os.path.relpath(os.path.join(root, file), artifact_path))

    if pytorch_files:
        flavors_found.append("pytorch")
        print(f"✅ PyTorch files found: {pytorch_files}")

    # PyFunc検出
    if os.path.exists(os.path.join(artifact_path, "python_model.pkl")):
        flavors_found.append("pyfunc")
        print("✅ PyFunc model detected (python_model.pkl found)")

    # TensorFlow検出
    tf_dirs = ['saved_model', 'tf_model']
    for tf_dir in tf_dirs:
        if os.path.exists(os.path.join(artifact_path, tf_dir)):
            flavors_found.append("tensorflow")
            print(f"✅ TensorFlow model detected ({tf_dir} found)")

    print(f"Detected flavors: {flavors_found}")
    return flavors_found

def find_pytorch_files(artifact_path):
    """PyTorchファイルの詳細検索"""
    pt_files = []

    for root, dirs, files in os.walk(artifact_path):
        for file in files:
            if file.endswith(('.pt', '.pth')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, artifact_path)
                file_size = os.path.getsize(file_path)

                pt_files.append({
                    'path': rel_path,
                    'full_path': file_path,
                    'size': file_size
                })

    if pt_files:
        print(f"✅ Found {len(pt_files)} PyTorch file(s):")
        for pt_file in pt_files:
            print(f"  - {pt_file['path']} ({format_file_size(pt_file['size'])})")

            # YOLOとして読み込みテスト
            try:
                from ultralytics import YOLO
                test_model = YOLO(pt_file['full_path'])
                print(f"    ✅ YOLO loading test: SUCCESS")
                return pt_file['full_path']
            except Exception as e:
                print(f"    ❌ YOLO loading test: FAILED ({e})")
    else:
        print("❌ No PyTorch (.pt/.pth) files found")

    return None

def main():
    """メイン実行関数"""
    # 設定（あなたの環境に合わせて変更）
    MLFLOW_MODEL_NAME = "disease-recognition-model"  # あなたのモデル名
    MLFLOW_TRACKING_URI = "https://mlflow.nasebanal.com"  # あなたのMLflow URI

    print(f"Target Model: {MLFLOW_MODEL_NAME}")
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")

    # 分析実行
    success = analyze_mlflow_model_structure(
        mlflow_model_name=MLFLOW_MODEL_NAME,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        version=None  # 最新バージョンを分析、特定バージョンなら数字を指定
    )

    if success:
        print(f"\n{'='*60}")
        print("✅ Analysis completed successfully")
        print("Check the output above to identify the issue")
    else:
        print(f"\n{'='*60}")
        print("❌ Analysis failed")

    print(f"{'='*60}")

if __name__ == "__main__":
    main()
