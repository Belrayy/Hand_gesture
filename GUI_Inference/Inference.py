import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.transforms as transforms # type: ignore
import torchvision.models as models # type: ignore
from PIL import Image # type: ignore
import json
import os
import numpy as np # type: ignore
import sys

# Add the TSM repository directory to the Python path
sys.path.append('/Pre_trained/Inference/temporal_shift_module/ops')
from Pre_trained.Inference.temporal_shift_module.ops.temporal_shift import make_temporal_shift  # Import from your temporal_shift.py

def load_model(checkpoint_path, num_classes, num_segments):
    # 1. Create base ResNet model
    base_model = models.resnet50(weights=False)  # Ensure backbone matches the checkpoint
    
    # 2. Add temporal shifts
    make_temporal_shift(base_model, num_segments, n_div=8, place='blockres')
    
    # 3. Replace the final layer for Jester
    feature_dim = base_model.fc.in_features
    base_model.fc = nn.Linear(feature_dim, num_classes)  # Now 27 classes
    
    # 4. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # 5. Adjust state_dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.base_model.'):
            name = k[len('module.base_model.'):]
        elif k.startswith('base_model.'):
            name = k[len('base_model.'):]
        elif k.startswith('module.'):
            name = k[len('module.'):]
        else:
            name = k
        name = name.replace('new_fc', 'fc').replace('.net', '')
        new_state_dict[name] = v
    
    # 6. Remove conflicting keys (add this block)
    keys = list(new_state_dict.keys())
    for key in keys:
        if key.startswith('fc.'):
            del new_state_dict[key]
    
    # 7. Load weights (strict=False to ignore missing keys)
    base_model.load_state_dict(new_state_dict, strict=False)
    base_model.eval()
    return base_model

def preprocess_frames(frames, transform):
    # Apply transforms and stack frames
    processed = [transform(frame) for frame in frames]
    return torch.stack(processed, dim=0)  # Shape: [T, C, H, W]

def sample_frames(video_dir, num_segments=8):
    # Load and sample frames from a directory of images
    frame_files = sorted(os.listdir(video_dir))
    total = len(frame_files)
    indices = np.linspace(0, total-1, num=num_segments, dtype=int)
    frames = []
    for i in indices:
        frame = Image.open(os.path.join(video_dir, frame_files[i])).convert('RGB')
        frames.append(frame)
    return frames

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    checkpoint_path = 'Pre_trained/PC.pth'  # Path to Jester checkpoint
    num_classes = 27  # 20BN-Jester has 27 classes
    num_segments = 8  # Verify if your Jester model uses 8 segments
    video_dir = 'data/archive/Validation/22'  
    label_file = 'Pre_trained/Inference/jester_labels.json'  # Jester label mappings
    
    # Load model
    model = load_model(checkpoint_path, num_classes, num_segments)
    model = model.to(device)
    
    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Sample and preprocess frames
    frames = sample_frames(video_dir, num_segments)
    input_tensor = preprocess_frames(frames, transform)
    
    # Reshape input for TSM (batch_size * num_segments, C, H, W)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.view(-1, 3, 224, 224).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Process output
    probs = torch.softmax(output[0], dim=0)
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    # Load labels
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    # Display results
    print("Top 5 Predictions:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"{i+1}. {labels[str(idx.item())]}: {prob.item()*100:.2f}%")

if __name__ == '__main__':
    main()