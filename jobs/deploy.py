"""Register a model to Unity Catalog and deploy to Model Serving.

Usage:
    python jobs/deploy.py --config_path configs/detection_yolos_config.yaml --run_id abc123 \
        --model_name catalog.schema.yolos_detection --endpoint_name yolos-detection-endpoint

    python jobs/deploy.py --config_path configs/detection_yolos_config.yaml --run_id abc123 \
        --model_name catalog.schema.yolos_detection --skip_test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path(sys.argv[0]).resolve() if sys.argv else Path(os.getcwd())

_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(description="Register and deploy a CV model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--model_name", type=str, default=None, help="Unity Catalog model name (catalog.schema.model)")
    parser.add_argument("--endpoint_name", type=str, default=None, help="Serving endpoint name")
    parser.add_argument("--workload_size", type=str, default=None, help="Endpoint workload size")
    parser.add_argument("--skip_test", action="store_true", help="Skip endpoint smoke test")
    parser.add_argument("--test_image", type=str, default=None, help="Path to test image for validation")
    args = parser.parse_args()

    from src.config.schema import load_config
    from src.serving.registration import register_model
    from src.serving.deployment import deploy_endpoint, wait_for_ready, test_endpoint

    config = load_config(args.config_path)

    model_name = args.model_name or config.serving.registered_model_name
    endpoint_name = args.endpoint_name or config.serving.endpoint_name
    workload_size = args.workload_size or config.serving.workload_size

    if not model_name:
        print("ERROR: --model_name is required (or set serving.registered_model_name in config)")
        sys.exit(1)

    # 1. Register model
    print("\n" + "=" * 60)
    print("MODEL REGISTRATION")
    print("=" * 60)
    reg_result = register_model(
        run_id=args.run_id,
        registered_model_name=model_name,
        task_type=config.model.task_type,
        aliases=["champion", "latest"],
        tags={"framework": "hf_trainer", "task": config.model.task_type},
        validate=True,
        test_image_path=args.test_image,
    )
    print(f"  Model: {reg_result['registered_model_name']}")
    print(f"  Version: {reg_result['model_version']}")
    print(f"  URI: {reg_result['model_uri']}")

    # 2. Deploy endpoint (if endpoint_name provided)
    if endpoint_name:
        print("\n" + "=" * 60)
        print("ENDPOINT DEPLOYMENT")
        print("=" * 60)
        deploy_result = deploy_endpoint(
            endpoint_name=endpoint_name,
            registered_model_name=model_name,
            model_version=str(reg_result["model_version"]),
            workload_size=workload_size,
            scale_to_zero=config.serving.scale_to_zero,
        )
        print(f"  Endpoint: {deploy_result['endpoint_name']}")
        print(f"  Status: {deploy_result['status']}")

        # 3. Wait for ready
        print("\nWaiting for endpoint to become READY...")
        wait_for_ready(endpoint_name)

        # 4. Smoke test
        if not args.skip_test:
            if args.test_image:
                print("\nRunning smoke test...")
                test_result = test_endpoint(
                    endpoint_name=endpoint_name,
                    test_image_path=args.test_image,
                )
                print(f"  Test result: {test_result}")
            else:
                print("\nSkipping smoke test (no --test_image provided)")
    else:
        print("\nNo endpoint_name — skipping deployment. Model registered only.")

    print("\nDone.")


if __name__ == "__main__":
    main()
