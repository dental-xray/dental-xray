def find_mlflow_size_limit(mlflow_tracking_uri, test_sizes_mb=[1, 2, 5, 10, 20, 50]):
    import mlflow
    import tempfile
    import os

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    print("🔍 Testing MLflow file size limits...")
    max_successful_size = 0

    for size_mb in test_sizes_mb:
        try:
            print(f"\n📊 Testing {size_mb}MB file...")

            with tempfile.NamedTemporaryFile(delete=False) as test_file:
                data = b'0' * (size_mb * 1024 * 1024)  # size_mb MB
                test_file.write(data)
                test_file_path = test_file.name

            try:
                with mlflow.start_run():
                    mlflow.log_artifact(test_file_path, artifact_path=f"size_test_{size_mb}mb")

                print(f"  ✅ {size_mb}MB: SUCCESS")
                max_successful_size = size_mb

            except Exception as upload_error:
                print(f"  ❌ {size_mb}MB: FAILED - {upload_error}")

                if "413" in str(upload_error) or "too large" in str(upload_error).lower():
                    print(f"  🎯 Found size limit between {max_successful_size}MB and {size_mb}MB")
                    break

            finally:
                os.unlink(test_file_path)

        except Exception as e:
            print(f"  ❌ {size_mb}MB: ERROR - {e}")

    print(f"\n📋 Results:")
    print(f"Maximum successful upload: {max_successful_size}MB")

    if max_successful_size > 0:
        print(f"Recommended chunk size: {max_successful_size // 2}MB")

    return max_successful_size

max_size = find_mlflow_size_limit("https://mlflow.nasebanal.com")


print(f"\nMaximum uploadable file size to MLflow server: {max_size}MB")
print(f"Recommended chunk size for uploads: {max_size // 2}MB")
