import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2 # type: ignore
from PIL import Image, ImageTk  # type: ignore
import threading
import time
import os
import torch  # type: ignore
import torch.nn as nn # type: ignore
import torchvision.transforms as transforms # type: ignore
import torchvision.models as models # type: ignore
import json
import numpy as np # type: ignore
from collections import deque
import sys

sys.path.append('/Pre_trained/Inference/temporal_shift_module/ops')
from Pre_trained.Inference.temporal_shift_module.ops.temporal_shift import make_temporal_shift  # type: ignore

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition System")
        self.root.geometry("1280x720")
        
        
        self.is_recording = False
        self.video_writer = None
        self.cap = None
        self.start_time = None
        self.output_dir = ""
        self.output_file = ""
        
        
        self.model = None
        self.labels = {}
        self.current_prediction = ""
        self.prediction_threshold = 0.7
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.frame_buffer = deque(maxlen=8)
        self.buffer_lock = threading.Lock()  # add a lock for frame_buffer
        self.prediction_interval = 5
        self.frame_counter = 0
        self.last_prediction_time = 0
        
        
        self.prediction_history = deque(maxlen=3)
        self.confidence_history = deque(maxlen=3)
        
        
        self.select_output_directory()
        self.load_model_and_labels()
        self.create_widgets()
        self.preview_video()
    
    def select_output_directory(self):
        self.output_dir = filedialog.askdirectory(title="Select Output Directory for Recordings")
        if not self.output_dir:
            messagebox.showwarning("Warning", "No directory selected. Using default location.")
            self.output_dir = os.path.join(os.path.expanduser("~"), "GestureRecordings")
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.status_var = tk.StringVar()
        self.status_var.set(f"Output directory: {self.output_dir}")
    
    def create_widgets(self):
        
        self.video_frame = ttk.Label(self.root)
        self.video_frame.pack(pady=10)
        
        
        controls_frame = ttk.Frame(self.root)
        controls_frame.pack(pady=10)
        
        
        self.record_btn = ttk.Button(
            controls_frame, 
            text="Start Recording", 
            command=self.toggle_recording
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.timer_label = ttk.Label(controls_frame, text="00:00:00")
        self.timer_label.pack(side=tk.LEFT, padx=5)
        
        
        self.prediction_frame = ttk.Frame(self.root)
        self.prediction_frame.pack(pady=10)
        
        self.prediction_label = ttk.Label(
            self.prediction_frame,
            text="Current Prediction: None",
            font=('Helvetica', 14)
        )
        self.prediction_label.pack()
        
        self.confidence_label = ttk.Label(
            self.prediction_frame,
            text="Confidence: 0%",
            font=('Helvetica', 12)
        )
        self.confidence_label.pack()
        
        
        self.action_label = ttk.Label(
            self.root,
            text="Action: Waiting for prediction...",
            font=('Helvetica', 12, 'bold'),
            foreground="blue"
        )
        self.action_label.pack(pady=10)
        
        
        ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(fill=tk.X, side=tk.BOTTOM)
    
    def load_model_and_labels(self):
        """Load the gesture recognition model and labels"""
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            checkpoint_path = 'Pre_trained/PC.pth'
            num_classes = 27
            num_segments = 8
            
            
            base_model = models.resnet101(weights="IMAGENET1K_V2")  # pretrained on ImageNet 
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
            self.model = base_model.to(self.device)
            
            
            test_input = torch.randn(8, 3, 224, 224).to(self.device)
            with torch.no_grad():
                test_output = self.model(test_input)
            print(f"Model test output shape: {test_output.shape}")
            
            
            with open('Pre_trained/Inference/jester_labels.json', 'r') as f:
                self.labels = json.load(f)
                
            self.status_var.set("Model loaded - Ready to record")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
    
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video device")
                return
        
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f"gesture_{timestamp}.avi")
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.output_file, fourcc, fps, (frame_width, frame_height))
        
        if not self.video_writer.isOpened():
            messagebox.showerror("Error", "Could not initialize video writer")
            return
        
        self.is_recording = True
        self.start_time = time.time()
        self.record_btn.config(text="Stop Recording")
        self.status_var.set(f"Recording to: {os.path.basename(self.output_file)}")
        
        self.update_timer()
    
    def stop_recording(self):
        self.is_recording = False
        self.record_btn.config(text="Start Recording")
        self.status_var.set(f"Recording saved to: {os.path.basename(self.output_file)}")
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    
    def update_timer(self):
        if self.is_recording:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.timer_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)
    
    def preview_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video device")
                return
        
        ret, frame = self.cap.read()
        if ret:
            
            if np.mean(frame) < 10:
                self.video_frame.after(30, self.preview_video)
                return
                
            
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = self.resize_with_aspect_ratio(display_frame, width=640, height=480)
            
            
            if self.is_recording:
                if self.video_writer:
                    self.video_writer.write(frame)
                
                with self.buffer_lock:
                    self.frame_buffer.append(frame.copy())
                
                self.frame_counter += 1
                
                with self.buffer_lock:
                    if len(self.frame_buffer) == 8:
                        if not hasattr(self, 'prediction_thread') or not self.prediction_thread.is_alive():
                            # pass a copy and clear the buffer immediately
                            frames_to_process = list(self.frame_buffer)
                            self.frame_buffer.clear()
                            self.prediction_thread = threading.Thread(
                                target=self.process_frames_for_prediction,
                                args=(frames_to_process,),
                                daemon=True
                            )
                            self.prediction_thread.start()
            
            
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        
        self.video_frame.after(30, self.preview_video)
    
    def process_frames_for_prediction(self, frames):
        """Process multiple frames for gesture prediction"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            processed_frames = []
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                tensor_frame = transform(pil_img)
                processed_frames.append(tensor_frame)
            
            input_tensor = torch.stack(processed_frames)  
            input_tensor = input_tensor.unsqueeze(0).to(self.device)  
            
            batch_size, num_segments, C, H, W = input_tensor.shape
            input_tensor = input_tensor.view(batch_size * num_segments, C, H, W)
            
            with torch.no_grad():
                output = self.model(input_tensor)
            
            output = output.view(batch_size, num_segments, -1).mean(dim=1)
            probs = torch.softmax(output, dim=1)
            top_prob, top_idx = torch.max(probs, 1)
            
            prediction = self.labels.get(str(top_idx.item()), "Unknown")
            confidence = top_prob.item() * 100
            
            self.root.after(0, self.update_prediction_ui, prediction, confidence)
            
        except Exception as e:
            print(f"Prediction error: {e}")
    
    def update_prediction_ui(self, prediction, confidence):
        """Always update the UI with the latest prediction"""
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)

        self.current_prediction = prediction
        self.prediction_label.config(text=f"Current Prediction: {prediction}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

        if confidence > self.prediction_threshold * 100:
            self.handle_prediction_action(prediction, confidence)
        else:
            self.action_label.config(text="Action: Low confidence", foreground="gray")
    
    def handle_prediction_action(self, prediction, confidence):
        """Take action based on the prediction"""
        action_text = f"Action: Detected {prediction} ({confidence:.1f}% confidence)"
        self.action_label.config(text=action_text)
        
        
        if "swipe left" in prediction.lower():
            self.action_label.config(foreground="red")
        elif "swipe right" in prediction.lower():
            self.action_label.config(foreground="green")
        else:
            self.action_label.config(foreground="blue")
    
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
        if self.is_recording:
            self.stop_recording()
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()