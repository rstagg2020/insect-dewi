import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import argparse
import math

# Add the project root to sys.path so we can import from models and utils
# Assuming this script is in <root>/demo_package/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import model architectures from the main repo
from models.dewi import dewi_resnet50 as dewi_standard
from models.transform_dewi import dewi_resnet50 as dewi_transformer

class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, init_scale=20.0):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor([init_scale]))
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(x, w) * self.scale
        return logits

def get_model(pipeline_type, num_classes):
    if pipeline_type == 'foc_tran':
        model = dewi_transformer(pth_url=None, pretrained=False)
        model.fc = CosineClassifier(model.fc.in_features, num_classes)
    elif pipeline_type == 'linear':
        model = dewi_standard(pth_url=None, pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif pipeline_type in ['focal', 'standard']:
        model = dewi_standard(pth_url=None, pretrained=False)
        model.fc = CosineClassifier(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_type}")
    return model

def load_weights(model, weights_path):
    # Check if the path is a directory (WandB Artifacts often download to a folder)
    if os.path.isdir(weights_path):
        # Look for .pth files in the directory
        pth_files = [f for f in os.listdir(weights_path) if f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"No .pth files found in {weights_path}")
        # Prioritize 'best_model.pth'
        if 'best_model.pth' in pth_files:
            weights_path = os.path.join(weights_path, 'best_model.pth')
        else:
            weights_path = os.path.join(weights_path, pth_files[0])

    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Check if it's a full checkpoint or just a state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Strip 'module.' prefix if it exists
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    return model

def main():
    parser = argparse.ArgumentParser(description="Insect Classification Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to the insect image")
    parser.add_argument("--pipeline", type=str, default="focal", choices=["focal", "linear", "foc_tran", "standard"], help="Which model pipeline to use")
    args = parser.parse_args()

    # Get the directory of the current script
    demo_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load class names (should be in the same folder or dataset_vt)
    classes_path = os.path.join(demo_dir, "classes.txt")
    if not os.path.exists(classes_path):
        # Fallback to project root if classes.txt isn't in demo_package
        classes_path = os.path.join(project_root, "dataset_vt/classes.txt")
        
    if not os.path.exists(classes_path):
        print(f"Error: classes.txt not found at {classes_path}.")
        return

    with open(classes_path, "r") as f:
        class_names = [line.strip().split(" ", 1)[1] for line in f.readlines()]
    num_classes = len(class_names)

    # 2. Setup Model
    # Try looking in the demo_package/weights/<pipeline> folder (artifact default)
    weights_path = os.path.join(demo_dir, "weights", args.pipeline)
    if not os.path.exists(weights_path):
        # Fallback to checking for a direct file weights/<pipeline>_best.pth
        weights_path = os.path.join(demo_dir, "weights", f"{args.pipeline}_best.pth")

    if not os.path.exists(weights_path):
        print(f"Error: Weights not found for {args.pipeline}. Please run download_weights.py first.")
        return

    print(f"Initializing {args.pipeline} model...")
    model = get_model(args.pipeline, num_classes)
    try:
        model = load_weights(model, weights_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Preprocess Image
    input_size = 384
    transform = transforms.Compose([
        transforms.Resize(input_size + 16),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(args.image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        _, logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]

    # 5. Get Top 5
    top5_prob, top5_idx = torch.topk(probs, 5)

    print("\n" + "="*45)
    print(f" Top 5 Predictions ({args.pipeline})")
    print("="*45)
    for i in range(5):
        name = class_names[top5_idx[i]]
        score = top5_prob[i].item() * 100
        print(f"{i+1}. {name:35} | {score:6.2f}%")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
