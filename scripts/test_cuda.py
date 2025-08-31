"""
CUDA 11.5 + Compute 6.1 disagnostic tool
Check if your environment is correctly set up for training YOLO models with GPU support.
"""

import torch
import sys
import os

def system_info():
    """システム情報表示"""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # CUDA環境変数確認
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    print(f"CUDA_HOME: {cuda_home}")
    print(f"LD_LIBRARY_PATH: {ld_library_path}")

def cuda_detection():
    """CUDA検出テスト"""
    print("\n=== CUDA Detection ===")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("❌ CUDA not detected by PyTorch")
        return False

    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    # 各GPU情報
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

    return True

def memory_test():
    """GPU メモリテスト"""
    print("\n=== GPU Memory Test ===")

    try:
        # 各GPUでメモリテスト
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            print(f"Testing GPU {i}...")

            # 小さなテンソル
            x = torch.rand(100, 100, device=f'cuda:{i}')
            print(f"  Small tensor OK on GPU {i}")

            # メモリ使用量確認
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"  Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")

            # メモリクリア
            del x
            torch.cuda.empty_cache()

        print("✅ Memory test passed")
        return True

    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def kernel_test():
    """CUDA カーネルテスト"""
    print("\n=== CUDA Kernel Test ===")

    try:
        # シンプルな演算テスト
        print("Testing basic operations...")
        a = torch.rand(1000, 1000, device='cuda')
        b = torch.rand(1000, 1000, device='cuda')

        # 加算
        c = a + b
        print("  Addition: OK")

        # 行列乗算
        d = torch.matmul(a, b)
        print("  Matrix multiplication: OK")

        # GPU間転送テスト（複数GPU場合）
        if torch.cuda.device_count() > 1:
            print("Testing multi-GPU operations...")
            a_gpu0 = torch.rand(100, 100, device='cuda:0')
            a_gpu1 = a_gpu0.to('cuda:1')
            print("  Multi-GPU transfer: OK")

        # CPU-GPU転送
        cpu_tensor = d.cpu()
        gpu_tensor = cpu_tensor.cuda()
        print("  CPU-GPU transfer: OK")

        print("✅ Kernel test passed")
        return True

    except RuntimeError as e:
        if "no kernel image is available" in str(e):
            print(f"❌ Kernel compatibility error: {e}")
            print("This indicates PyTorch CUDA version doesn't match your GPU architecture")
            return False
        else:
            print(f"❌ Kernel test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected kernel test error: {e}")
        return False

def model_test():
    """Neural Networkモデルテスト"""
    print("\n=== Neural Network Model Test ===")

    try:
        import torch.nn as nn

        # シンプルなモデル
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        print("Testing model transfer to GPU...")
        model = model.cuda()
        print("  Model transfer: OK")

        # ダミーデータで推論
        print("Testing inference...")
        x = torch.rand(32, 784, device='cuda')
        output = model(x)
        print("  Inference: OK")

        # 逆伝播テスト
        print("Testing backpropagation...")
        loss_fn = nn.CrossEntropyLoss()
        target = torch.randint(0, 10, (32,), device='cuda')
        loss = loss_fn(output, target)
        loss.backward()
        print("  Backpropagation: OK")

        print("✅ Model test passed")
        return True

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def yolo_compatibility():
    """YOLO互換性テスト"""
    print("\n=== YOLO Compatibility Test ===")

    try:
        from ultralytics import YOLO

        print("Loading smallest YOLO model...")
        model = YOLO('yolov8n.pt')  # 最小モデル

        print("Testing CPU inference first...")
        import numpy as np
        dummy_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

        # CPU推論
        results = model(dummy_img, device='cpu', verbose=False)
        print("  CPU inference: OK")

        # GPU推論テスト
        print("Testing GPU inference...")
        results = model(dummy_img, device='cuda', verbose=False)
        print("  GPU inference: OK")

        # モデル情報
        print("Testing model info...")
        model.info(verbose=False)
        print("  Model info: OK")

        print("✅ YOLO compatibility passed")
        return True

    except Exception as e:
        print(f"❌ YOLO compatibility failed: {e}")
        print("Try running with device='cpu' for now")
        return False

def recommendations():
    """推奨事項"""
    print("\n=== Recommendations ===")

    # PyTorchバージョン確認
    torch_version = torch.__version__
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "None"

    print(f"Current PyTorch: {torch_version}")
    print(f"Current PyTorch CUDA: {cuda_version}")

    print("\nFor CUDA 11.5 + Compute 6.1, recommended installations:")
    print("Option 1 (Stable):")
    print("  pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116")

    print("\nOption 2 (Newer):")
    print("  pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118")

    print("\nIf problems persist:")
    print("  - Use device='cpu' in your training code")
    print("  - Check CUDA_HOME and LD_LIBRARY_PATH environment variables")
    print("  - Consider updating CUDA toolkit to 11.6 or 11.8")

def main():
    """メイン診断関数"""
    print("CUDA 11.5 + Compute Capability 6.1 Diagnostic Tool")
    print("=" * 60)

    # 段階的診断
    system_info()

    if not cuda_detection():
        recommendations()
        return

    if not memory_test():
        recommendations()
        return

    if not kernel_test():
        print("\n⚠️  Kernel test failed - this is likely the root cause")
        recommendations()
        return

    if not model_test():
        recommendations()
        return

    yolo_success = yolo_compatibility()

    if yolo_success:
        print("\n🎉 All tests passed! Your YOLO training should work.")
    else:
        print("\n⚠️  YOLO specific issues detected. Try CPU mode for now.")

    recommendations()

if __name__ == "__main__":
    main()
