import wandb
import os
import argparse

def download_artifact(entity, project, name, version="latest", output_path="weights"):
    """Downloads a model artifact from WandB."""
    api = wandb.Api()
    artifact_path = f"{entity}/{project}/{name}:{version}"
    print(f"Downloading artifact: {artifact_path}...")
    
    try:
        artifact = api.artifact(artifact_path)
        datadir = artifact.download(root=output_path)
        print(f"Successfully downloaded to {datadir}")
        return True
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download model weights from WandB Artifacts")
    parser.add_argument("--entity", type=str, default="rstagg-university-of-vermont", help="WandB entity (username or team)")
    parser.add_argument("--project", type=str, default="dewi-insect-classification", help="WandB project name")
    args = parser.parse_args()

    os.makedirs("weights", exist_ok=True)

    # Dictionary of artifacts to download
    # Key: Pipeline name, Value: Artifact name in WandB
    artifacts = {
        "focal": "focal-best-model",
        "linear": "linear-best-model",
        "foc_tran": "foc_tran-best-model",
        "standard": "standard-best-model"
    }

    print(f"Starting downloads for project: {args.entity}/{args.project}\n")
    
    for pipeline, artifact_name in artifacts.items():
        print(f"--- Pipeline: {pipeline} ---")
        success = download_artifact(args.entity, args.project, artifact_name, output_path=f"weights/{pipeline}")
        if not success:
            print(f"Note: Ensure you have logged an artifact named '{artifact_name}' to WandB.")

    print("\nDownload process complete.")

if __name__ == "__main__":
    main()
