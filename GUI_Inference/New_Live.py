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


# Hagrid v2 gesture labels
GESTURE_LABELS = [
    "call", "dislike", "fist", "four", "like", "mute", 
    "ok", "one", "palm", "peace", "peace_inverted", 
    "rock", "stop", "stop_inverted", "three", "three2", 
    "two_up", "two_up_inverted"
]

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Recognition System (Hagrid v2)")
        self.root.geometry("1280x720")
        
        # Video capture and recording
        self.is_recording = False
        self.video_writer = None
        self.cap = None
        self.start_time = None
        self.output_dir = ""
        self.output_file = ""
        
        # Model and prediction
        self.model = None
        self.current_prediction = ""
        self.prediction_threshold = 0.7
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.frame_buffer = deque(maxlen=16)  # Buffer for temporal analysis
        self.buffer_lock = threading.Lock()
        
        # UI state
        self.prediction_history = deque(maxlen=3)
        self.confidence_history = deque(maxlen=3)
        
        # Initialize components
        self.select_output_directory()
        self.load_model()
        self.create_widgets()
        self.preview_video()
    
    def select_output_directory(self):
        """Let user select where to save recordings"""
        self.output_dir = filedialog.askdirectory(title="Select Output Directory for Recordings")
        if not self.output_dir:
            messagebox.showwarning("Warning", "No directory selected. Using default location.")
            self.output_dir = os.path.join(os.path.expanduser("~"), "GestureRecordings")
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.status_var = tk.StringVar()
        self.status_var.set(f"Output directory: {self.output_dir}")
    
    def create_widgets(self):
        """Create the GUI components"""
        # Video display
        self.video_frame = ttk.Label(self.root)
        self.video_frame.pack(pady=10)
        
        # Control buttons
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
        
        # Prediction display
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
        
        # Action feedback
        self.action_label = ttk.Label(
            self.root,
            text="Action: Waiting for prediction...",
            font=('Helvetica', 12, 'bold'),
            foreground="blue"
        )
        self.action_label.pack(pady=10)
        
        # Status bar
        ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(fill=tk.X, side=tk.BOTTOM)
    
    def load_model(self):
        """Load the YOLOv10x model from local file"""
        try:
            self.status_var.set("Loading YOLOv10x model...")
            self.root.update()
        
            # Path to your downloaded model file
            model_path = 'Pre_trained/YOLO.pt'  # Update this path
        
            # Verify model file exists
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found at: {model_path}")
                self.root.destroy()
                return
        
            # Load the YOLOv5 model with custom weights
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                   path=model_path, 
                                   force_reload=True)
        
            # Set device and evaluation mode
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
        
            self.status_var.set("YOLOv10x model loaded - Ready to record")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        self.root.destroy()
    
    def toggle_recording(self):
        """Toggle recording state"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start video recording"""
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
        """Stop video recording"""
        self.is_recording = False
        self.record_btn.config(text="Start Recording")
        self.status_var.set(f"Recording saved to: {os.path.basename(self.output_file)}")
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    
    def update_timer(self):
        """Update recording timer"""
        if self.is_recording:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.timer_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)
    
    def predict_gesture(self, frame):
        """Predict gesture from frame using YOLOv10x"""
        try:
            # Convert frame to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(frame_rgb)
            
            # Get predictions
            preds = results.pandas().xyxy[0]
            
            if len(preds) > 0:
                # Get prediction with highest confidence
                best_pred = preds.iloc[0]
                gesture = best_pred['name']
                confidence = best_pred['confidence']
                
                return gesture, confidence * 100
            else:
                return "No gesture", 0
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0
    
    def process_frames_for_prediction(self, frame):
        """Process frame and update prediction"""
        prediction, confidence = self.predict_gesture(frame)
        self.root.after(0, self.update_prediction_ui, prediction, confidence)
    
    def update_prediction_ui(self, prediction, confidence):
        """Update UI with prediction"""
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
        """Handle prediction actions"""
        action_text = f"Action: Detected {prediction} ({confidence:.1f}% confidence)"
        self.action_label.config(text=action_text)
        
        if "dislike" in prediction.lower():
            self.action_label.config(foreground="red")
        elif "like" in prediction.lower():
            self.action_label.config(foreground="green")
        else:
            self.action_label.config(foreground="blue")
    
    def preview_video(self):
        """Preview video from camera"""
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
                
                # Process every 5th frame for prediction
                if len(self.frame_buffer) == 5:
                    if not hasattr(self, 'prediction_thread') or not self.prediction_thread.is_alive():
                        frame_to_process = self.frame_buffer[-1]  # Use most recent frame
                        self.prediction_thread = threading.Thread(
                            target=self.process_frames_for_prediction,
                            args=(frame_to_process,),
                            daemon=True
                        )
                        self.prediction_thread.start()
            
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        
        self.video_frame.after(30, self.preview_video)
    
    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        """Resize image while maintaining aspect ratio"""
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
        """Handle window closing"""
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