import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2 # type: ignore # type: ignore
from PIL import Image, ImageTk # type: ignore
import threading
import time
import os
import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.transforms as transforms # type: ignore
import torchvision.models as models # type: ignore
import json
import numpy as np # type: ignore
import tempfile
import sys

sys.path.append('/Pre_trained/Inference/temporal_shift_module/ops')
from Pre_trained.Inference.temporal_shift_module.ops.temporal_shift import make_temporal_shift

class VideoAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Gesture Recognition")
        self.root.geometry("1280x720")
        
        # Video variables
        self.cap = None
        self.current_video_path = ""
        self.model = None
        self.labels = {}
        
        self.create_widgets()
        self.load_model_and_labels()
        
    def create_widgets(self):
        # Video display
        self.video_frame = ttk.Label(self.root)
        self.video_frame.pack(pady=10)
        
        # Controls
        controls_frame = ttk.Frame(self.root)
        controls_frame.pack(pady=10)
        
        ttk.Button(
            controls_frame,
            text="Select Video",
            command=self.select_video
        ).pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = ttk.Button(
            controls_frame,
            text="Predict Gesture",
            command=self.start_prediction,
            state=tk.DISABLED
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.results_text = tk.Text(self.root, height=8, width=50)
        self.results_text.pack(pady=10)
        self.results_text.insert(tk.END, "Prediction results will appear here...")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(fill=tk.X, side=tk.BOTTOM)
    
    def load_model_and_labels(self):
        # Load gesture recognition model
        self.status_var.set("Loading model...")
        checkpoint_path = 'Pre_trained/PC.pth'
        num_classes = 27
        num_segments = 8
        
        # Model loading code from Inference.py
        base_model = models.resnet50(weights=False)    
        make_temporal_shift(base_model, num_segments, n_div=8, place='blockres')
        feature_dim = base_model.fc.in_features
        base_model.fc = nn.Linear(feature_dim, num_classes)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
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
            
        keys = list(new_state_dict.keys())
        for key in keys:
            if key.startswith('fc.'):
                del new_state_dict[key]
                
        base_model.load_state_dict(new_state_dict, strict=False)
        base_model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = base_model.to(self.device)
        
        # Load labels
        with open('Pre_trained/Inference/jester_labels.json', 'r') as f:
            self.labels = json.load(f)
            
        self.status_var.set("Model loaded")
    
    def select_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if file_path:
            self.current_video_path = file_path
            self.predict_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Selected: {os.path.basename(file_path)}")
            self.show_video_preview()
    
    def show_video_preview(self):
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.current_video_path)
        self.update_preview()
    
    def update_preview(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.resize_with_aspect_ratio(frame, width=640, height=360)
            
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
            self.video_frame.after(25, self.update_preview)
        else:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.current_video_path)
    
    def start_prediction(self):
        self.status_var.set("Processing video...")
        threading.Thread(target=self.process_video).start()
    
    def process_video(self):
        # Extract frames
        temp_dir = tempfile.mkdtemp()
        self.extract_frames(temp_dir)
        
        # Preprocess frames
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        frames = self.sample_frames(temp_dir)
        input_tensor = torch.stack([transform(frame) for frame in frames], dim=0)
        input_tensor = input_tensor.unsqueeze(0).view(-1, 3, 224, 224).to(self.device)
        
        # Run prediction
        with torch.no_grad():
            output = self.model(input_tensor)
        
        probs = torch.softmax(output[0], dim=0)
        top5_probs, top5_indices = torch.topk(probs, 5)
        
        # Update results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Top Predictions:\n")
        best_prob = top5_probs[0]
        best_idx = top5_indices[0]
        self.results_text.insert(tk.END, 
                                 f"- {self.labels[str(best_idx.item())]}: {best_prob.item()*100:.1f}% confidence")
        
        self.status_var.set("Prediction complete")
    
    def extract_frames(self, output_dir):
        cap = cv2.VideoCapture(self.current_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames-1, 8, dtype=int)
        
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(output_dir, f"frame_{i}.jpg"), frame)
        cap.release()
    
    def sample_frames(self, video_dir):
        frame_files = sorted(os.listdir(video_dir))
        return [Image.open(os.path.join(video_dir, f)).convert('RGB') for f in frame_files]
    
    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is not None:
            r = width / w
            dim = (width, int(h * r))
        else:
            r = height / h
            dim = (int(w * r), height)
        return cv2.resize(image, dim, interpolation=inter)
    
    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()