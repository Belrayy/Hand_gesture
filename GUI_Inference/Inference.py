import torch # type: ignore
import torch.nn as nn  # type: ignore
import torchvision.transforms as transforms  # type: ignore
from PIL import Image  # type: ignore
import json
import os
import numpy as np  # type: ignore
import sys

# Add the TSM repository directory to the Python path
sys.path.append('/Pre_trained/Inference/temporal-shift-module')
from Pre_trained.Inference.temporal_shift_module.ops.temporal_shift import TemporalShift

def load_model(checkpoint_path, num_classes, num_segments):
    # Initialize the TSM model
    model = TemporalShift(num_classes, n_segment=num_segments, modality='RGB',
                base_model='resnet50', consensus_type='avg',
                dropout=0.5, partial_bn=False, pretrain='imagenet',
                is_shift=True, shift_div=8, shift_place='blockres',
                non_local=False, temporal_pool=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Adjust for DataParallel wrapping
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # Remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    model.eval()  # Set to evaluation mode
    return model

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
    checkpoint_path = 'Pre_trained/PC.pth'
    num_classes = 400  # Kinetics-400 classes
    num_segments = 8
    video_dir = 'data/archive/Test/2'  # Directory containing extracted frames
    label_file = 'Pre_trained/Inference/kinetics_labels.json'  # Path to label mappings
    
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
    input_tensor = preprocess_frames(frames, transform).to(device)
    
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