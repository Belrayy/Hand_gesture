import os
import cv2
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import mediapipe as mp  # type: ignore
from threading import Thread
import math

class VideoRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prometheus")
        
        try:
            subdirectory_icon_path = "../Icon/Img/icon.ico"
            icon_path = os.path.join(os.path.dirname(__file__), subdirectory_icon_path)
            self.root.iconbitmap(icon_path)
        except:
            try:
                subdirectory_icon_path = "../Icon/Img/Hand_gesture_app.png"
                icon_path = os.path.join(os.path.dirname(__file__), subdirectory_icon_path)
                img = tk.PhotoImage(file=icon_path)
                self.root.tk.call('wm', 'iconphoto', self.root._w, img)
            except:
                pass

        self.recording = False
        self.start_time = None
        self.output_folder = ""
        self.cap = None
        self.out = None
        self.hands = None

        self.paused = False  
        self.accumulated_time = 0  
        self.last_pause_time = 0
        
        
        self.output_folder = self.show_initial_folder_dialog()
        if not self.output_folder:
            self.root.destroy()
            return
        
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        
        self.create_widgets()
        
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video device") # type: ignore
            self.root.destroy()
            return
        
        
        self.update_preview()
    
    def show_initial_folder_dialog(self):
        """Show folder selection dialog before main window appears"""
        temp_root = tk.Tk()
        temp_root.withdraw()  
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subdirectory = "data"
        initial_dir = os.path.join(script_dir, subdirectory)
        
        folder = filedialog.askdirectory(
            title="Select Output Folder for Recordings",
            initialdir=initial_dir,
            mustexist=True
        )
        
        if not folder:
            messagebox.showwarning("Warning", "You must select a folder to continue") # type: ignore
            return ""
        
        
        recordings_folder = os.path.join(folder, "Gesture_Recordings")
        os.makedirs(recordings_folder, exist_ok=True)
        
        temp_root.destroy()
        return recordings_folder
    
    def create_widgets(self):
        background_color = "#005f73"
        self.root.configure(bg=background_color)  
    
        self.preview_label = tk.Label(self.root, bg=background_color)
        self.preview_label.pack()
    
        status_frame = tk.Frame(self.root, bg=background_color)
        status_frame.pack(pady=10)
    
        self.timer_label = tk.Label(status_frame, text="00:00:00", font=('Arial', 14), bg=background_color)
        self.timer_label.pack(side=tk.LEFT, padx=10)
    
        self.status_label = tk.Label(status_frame, text="Ready", font=('Arial', 14), bg=background_color)
        self.status_label.pack(side=tk.LEFT, padx=10)
    
        self.gesture_label = tk.Label(self.root, text="Gesture: None", font=('Arial', 16), bg=background_color)
        self.gesture_label.pack(pady=5)
    
        button_frame = tk.Frame(self.root, bg=background_color)
        button_frame.pack(pady=10)
    
        self.record_button = tk.Button(
            button_frame, 
            text="Start/Pause", 
            command=self.toggle_recording,
            bg="#4CAF50",
            fg="white",
            font=('Arial', 12),
            width=15
        )
        self.record_button.pack(side=tk.LEFT, padx=10)
    
        # Add Full Stop button
        self.full_stop_button = tk.Button(
            button_frame, 
            text="Full Stop", 
            command=self.full_stop_recording,
            bg="#f44336",
            fg="white",
            font=('Arial', 12),
            width=15,
            state=tk.DISABLED  # Disabled by default
        )
        self.full_stop_button.pack(side=tk.LEFT, padx=10)
    
        tk.Button(
            button_frame, 
            text="Exit", 
            command=self.close_app,
            bg="#333333",
            fg="white",
            font=('Arial', 12),
            width=15
        ).pack(side=tk.LEFT)
    
    def select_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder = folder_selected
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, folder_selected)
    
    def toggle_recording(self):
        if not self.recording:
            # Start or resume recording
            self.start_recording()
            self.record_button.config(text="Pause")
            self.full_stop_button.config(state=tk.NORMAL)  # Enable full stop button
            self.status_label.config(text="Recording", fg="red")
        else:
            if not self.paused:
                # Pause recording
                self.pause_recording()
                self.record_button.config(text="Resume")
                self.status_label.config(text="Paused", fg="orange")
            else:
                # Resume recording
                self.resume_recording()
                self.record_button.config(text="Pause")
                self.status_label.config(text="Recording", fg="red")

    def full_stop_recording(self):
        """Completely stop and finalize the recording"""
        self.stop_recording()
        self.record_button.config(text="Start/Pause")
        self.full_stop_button.config(state=tk.DISABLED)  # Disable full stop button
        self.status_label.config(text="Ready", fg="black")
        self.timer_label.config(text="00:00:00")
    
        # Show confirmation message
        tk.messagebox.showinfo("Recording Saved", "The recording has been saved successfully.")
    

    def start_recording(self):
        if not hasattr(self, 'output_file') or self.output_file is None:
            # Only create new file if we don't have one
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = os.path.join(self.output_folder, f"recording_{timestamp}.avi")
        
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
        
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.output_file, fourcc, fps, (frame_width, frame_height))

        self.recording = True
        self.paused = False
        self.start_time = time.time() - self.accumulated_time
        self.update_timer()

    def pause_recording(self):
        self.paused = True
        self.recording = False
        self.accumulated_time += time.time() - self.start_time
        if self.out:
            self.out.release()  # Release the writer when pausing

    def resume_recording(self):
        if self.out is None:
            # Recreate the writer if needed
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.output_file, fourcc, fps, (frame_width, frame_height))

        self.recording = True
        self.paused = False
        self.start_time = time.time() - self.accumulated_time

    def stop_recording(self):
        """Internal method to stop recording and clean up"""
        self.recording = False
        self.paused = False
        self.accumulated_time = 0
        if self.out:
            self.out.release()
            self.out = None
        self.output_file = None
    
    def update_timer(self):
        if self.recording:
            elapsed = self.accumulated_time + (time.time() - self.start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.timer_label.config(text=f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            self.root.after(1000, self.update_timer)
    
    def update_preview(self):
        ret, frame = self.cap.read()
        if ret:
            
            frame = cv2.flip(frame, 1)
            
            
            processed_frame, gesture = self.process_frame(frame)
            
            
            self.update_gesture_display(gesture)
            
            
            if self.recording and self.out:
                self.out.write(cv2.flip(frame, 1))  
            
            
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(processed_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        
        self.root.after(10, self.update_preview)
    
    def process_frame(self, frame):
        gesture = "None"
        
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        results = self.hands.process(rgb_frame)
        self.hands.last_results = results
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                
                gesture = self.detect_gesture(hand_landmarks)
        
        
        return frame, gesture
    
    def detect_gesture(self, hand_landmarks):
        
        handedness = "Right"
        if hasattr(self.hands, 'last_results') and hasattr(self.hands.last_results, 'multi_handedness'):
            
            for idx, hand in enumerate(self.hands.last_results.multi_hand_landmarks):
                if hand == hand_landmarks:
                    handedness = self.hands.last_results.multi_handedness[idx].classification[0].label
                    break

        landmarks = hand_landmarks.landmark

        def finger_extended(tip, pip, mcp):
            return tip.y < pip.y < mcp.y

        def thumb_extended():
            vx = landmarks[5].x - landmarks[0].x
            vy = landmarks[5].y - landmarks[0].y
            ux = landmarks[4].x - landmarks[0].x
            uy = landmarks[4].y - landmarks[0].y
            cross = vx * uy - vy * ux
            if handedness == "Right":
                return cross < 0
            else:
                return cross > 0

        fingers = []
        fingers.append(thumb_extended())
        fingers.append(finger_extended(landmarks[8], landmarks[6], landmarks[5]))
        fingers.append(finger_extended(landmarks[12], landmarks[10], landmarks[9]))
        fingers.append(finger_extended(landmarks[16], landmarks[14], landmarks[13]))
        fingers.append(finger_extended(landmarks[20], landmarks[18], landmarks[17]))

        if all(fingers):
            return "Open Hand"
        if not any(fingers):
            return "Fist"
        if fingers == [True, False, False, False, False]:
            return "Thumbs Up"
        if fingers == [False, True, True, False, False]:
            return "Peace Sign"
        if fingers == [False, True, False, False, False]:
            return "Pointing"
        if fingers == [True, False, False, False, True]:
            return "Hang Loose"
        if fingers == [False, True, True, True, True]:
            return "Number Four"
        if fingers == [False, True, True, True, False]:
            return "Number Three"
        if fingers == [False, True, False, False, True]:
            return "Two"
        if fingers == [True, False, True, False, True]:
            return "Spider-Man"
        if fingers == [True, False, False, True, True]:
            return "Rock-on"
        if fingers == [False, True, True, False, False] and self._thumb_touching_index(landmarks):
            return "Okay"
        if fingers == [True, False, False, False, False] and self._pinky_extended(landmarks):
            return "Call Me"
        if fingers == [False, True, False, False, False] and not finger_extended(landmarks[12], landmarks[10], landmarks[9]):
            return "Closed Fist with Pointing Index"
        if fingers == [True, True, True, True, True] and self._flat_palm(landmarks):
            return "Flat Hand"
        if fingers == [False, True, True, False, False]:
            return "Victory"
        if fingers == [False, True, False, False, False] and self._index_extended_only(landmarks):
            return "Gun"
    
        return "Unknown"
    
    def update_gesture_display(self, gesture):
        self.gesture_label.config(text=f"Gesture: {gesture}")
        
        
        if gesture == "Open Hand":
            self.gesture_label.config(fg="green")
        elif gesture == "Fist":
            self.gesture_label.config(fg="red")
        elif gesture == "Thumbs Up":
            self.gesture_label.config(fg="blue")
        elif gesture == "Peace Sign":
            self.gesture_label.config(fg="purple")
        elif gesture == "Pointing":
            self.gesture_label.config(fg="orange")
        else:
            self.gesture_label.config(fg="black")
    

    def close_app(self):
        # Make sure to stop properly when closing
        if self.recording or self.paused:
            self.stop_recording()
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    
    
    window_width = 1200
    window_height = 720
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    app = VideoRecorderApp(root)
    root.mainloop()