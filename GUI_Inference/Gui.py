import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2 # type: ignore
from PIL import Image, ImageTk  # type: ignore
import threading
import time
import os

class VideoRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Recorder")
        self.root.geometry("800x600")
        
        # Video recording variables
        self.is_recording = False
        self.video_writer = None
        self.cap = None
        self.start_time = None
        self.output_file = ""
        
        self.create_widgets()
        
        self.preview_video()
    
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
        
        ttk.Button(
            controls_frame, 
            text="Select Output File", 
            command=self.select_output_file
        ).pack(side=tk.LEFT, padx=5)
        
        self.timer_label = ttk.Label(controls_frame, text="00:00:00")
        self.timer_label.pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(fill=tk.X, side=tk.BOTTOM)
    
    def select_output_file(self):
        if self.is_recording:
            messagebox.showerror("Error", "Cannot change output file while recording")
            return
            
        default_filename = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        if file_path:
            self.output_file = file_path
            self.status_var.set(f"Output file: {os.path.basename(self.output_file)}")
    
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        if not self.output_file:
            messagebox.showerror("Error", "Please select an output file first")
            return
            
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video device")
                return
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0  #Could be 60
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # For AVI / Can change video format
        if self.output_file.lower().endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.resize_with_aspect_ratio(frame, width=640)
            
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
            
            if self.is_recording and self.video_writer:
                original_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(original_frame)
        
        self.video_frame.after(10, self.preview_video)
    
    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        
        if width is None and height is None:
            return image
        
        if width is not None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            r = height / float(h)
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
    app = VideoRecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()