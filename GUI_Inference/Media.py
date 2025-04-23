import os
from tkinter import messagebox
import cv2
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import mediapipe as mp  # type: ignore
from threading import Thread
import math
import webbrowser

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
            messagebox.showwarning("Warning", "You must select a folder to continue")
            return ""
        
        recordings_folder = os.path.join(folder, "Gesture_Recordings")
        os.makedirs(recordings_folder, exist_ok=True)
        
        temp_root.destroy()
        return recordings_folder
    
    def create_widgets(self):
        background_color = "#005f73"
        self.root.configure(bg=background_color)  
        
        # Create a container frame for the main content
        self.main_container = tk.Frame(self.root, bg=background_color)
        self.main_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Sidebar
        self.sidebar_icon = tk.Button(
            self.root, text="☰", font=("Arial", 18), bg="#333333", fg="white",
            command=self.toggle_sidebar
        )
        self.sidebar_icon.place(x=10, y=10, width=40, height=40)

        self.sidebar_frame = tk.Frame(self.root, bg=background_color, width=200, height=720)
        self.sidebar_frame.place(x=-200, y=0)

        # Sidebar options
        self.sidebar_buttons = []
        for idx, option in enumerate(["Home", "Gestures", "About"]):
            btn = tk.Button(
                self.sidebar_frame, text=option, font=("Arial", 14),
                bg="#4CAF50" if option == "Home" else "#f9c74f" if option == "Gestures" else "#f44336",
                fg="white", width=10, height=2,
                command=lambda: (
                    self.show_gestures_page() if option == "Gestures" else
                    self.show_about_page() if option == "About" else
                    self.show_main_content()
                )
)
            btn.place(x=10, y=60 + idx*70)
            self.sidebar_buttons.append(btn)
    
        # Main content frame (for camera preview and controls)
        self.main_content_frame = tk.Frame(self.main_container, bg=background_color)
        self.main_content_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_label = tk.Label(self.main_content_frame, bg=background_color)
        self.preview_label.pack()
    
        self.status_frame = tk.Frame(self.main_content_frame, bg=background_color)
        self.status_frame.pack(pady=10)
    
        self.timer_label = tk.Label(self.status_frame, text="00:00:00", font=('Arial', 14), bg=background_color)
        self.timer_label.pack(side=tk.LEFT, padx=10)
    
        self.status_label = tk.Label(self.status_frame, text="Ready", font=('Arial', 14), bg=background_color)
        self.status_label.pack(side=tk.LEFT, padx=10)
    
        self.gesture_label = tk.Label(self.main_content_frame, text="Gesture: None", font=('Arial', 16), bg=background_color)
        self.gesture_label.pack(pady=5)
    
        self.button_frame = tk.Frame(self.main_content_frame, bg=background_color)
        self.button_frame.pack(pady=10)
    
        self.record_button = tk.Button(
            self.button_frame, 
            text="Start/Pause", 
            command=self.toggle_recording,
            bg="#4CAF50",
            fg="white",
            font=('Arial', 12),
            width=15
        )
        self.record_button.pack(side=tk.LEFT, padx=10)
    
        self.full_stop_button = tk.Button(
            self.button_frame, 
            text="Full Stop", 
            command=self.full_stop_recording,
            bg="#f44336",
            fg="white",
            font=('Arial', 12),
            width=15,
            state=tk.DISABLED
        )
        self.full_stop_button.pack(side=tk.LEFT, padx=10)
    
        self.exit_button = tk.Button(
            self.button_frame, 
            text="Exit", 
            command=self.close_app,
            bg="#333333",
            fg="white",
            font=('Arial', 12),
            width=15
        )
        self.exit_button.pack(side=tk.LEFT)

        # About page frame (initially hidden)
        self.about_frame = tk.Frame(self.main_container, bg=background_color)
        tk.Label(self.about_frame, text="About Us", font=("Arial", 32), bg=background_color, fg="white").pack(pady=20)
        for widget in self.about_frame.winfo_children():
            widget.destroy()

        tk.Label(self.about_frame, text="About Us", font=("Arial", 32), bg=background_color, fg="white").pack(pady=10)

        left_group = tk.Frame(self.about_frame, bg=background_color)
        left_group.pack(pady=10, padx=20, fill=tk.X)

        left_qr_image = Image.open("../Icon/QR/git.png")
        left_qr_photo = ImageTk.PhotoImage(left_qr_image)
        left_qr_label = tk.Label(left_group, image=left_qr_photo, bg=background_color)
        left_qr_label.image = left_qr_photo  # Keep a reference
        left_qr_label.pack()  # Default side is TOP

        left_link = tk.Label(left_group, text="Visit the GitHub Repo", font=("Arial", 14, "underline"), fg="cyan", bg=background_color, cursor="hand2")
        left_link.pack(pady=(5, 0)) # Add some padding above the link
        left_link.bind("<Button-1>", lambda e: open_link("https://github.com/Belrayy/Hand_gesture"))

        # Frame for the right QR code and link
        right_group = tk.Frame(self.about_frame, bg=background_color)
        right_group.pack(pady=10, padx=20, fill=tk.X)

        right_qr_image = Image.open("../Icon/QR/Web.png")
        right_qr_photo = ImageTk.PhotoImage(right_qr_image)
        right_qr_label = tk.Label(right_group, image=right_qr_photo, bg=background_color)
        right_qr_label.image = right_qr_photo
        right_qr_label.pack() # Default side is TOP

        right_link = tk.Label(right_group, text="Visit Our Web Site", font=("Arial", 14, "underline"), fg="cyan", bg=background_color, cursor="hand2")
        right_link.pack(pady=(5, 0)) # Add some padding above the link
        right_link.bind("<Button-1>", lambda e: open_link("https://belrayy.github.io/Hand_gesture/Web_site/index.html"))

        tk.Label(self.about_frame, text="We’re two passionate engineering students building Prometheus as our end-of-year project. \n Prometheus is more than just a project, it’s our vision of intuitive, touchless control in action.", font=("Arial", 12), bg=background_color, fg="white").pack(pady=10)

        self.gestures_frame = tk.Frame(self.main_container, bg=background_color)
        tk.Label(self.gestures_frame, text="Gesture Guide", font=("Arial", 32), bg=background_color, fg="white").pack(pady=20)
        gesture_container = tk.Frame(self.gestures_frame, bg=background_color)
        gesture_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        gestures_info = [
            ("Open Hand", "All fingers extended"),
            ("Fist", "All fingers closed"),
            ("Thumbs Up", "Only thumb extended"),
            ("Peace Sign", "Index and middle fingers extended"),
            ("Pointing", "Only index finger extended"),
            ("Hang Loose", "Thumb and pinky extended"),
            ("Number Four", "All fingers except thumb extended"),
            ("Number Three", "Index, middle, and ring fingers extended"),
            ("Two", "Index and pinky fingers extended"),
            ("Spider-Man", "Thumb, index, and pinky fingers extended"),
            ("Rock-on", "Thumb, ring, and pinky fingers extended"),
            ("Okay", "Thumb and index finger touching"),
            ("Call Me", "Thumb extended and pinky raised"),
            ("Victory", "Index and middle fingers extended (V sign)"),
            ("Gun", "Index finger extended and thumb up")
        ]
    
        for i, (gesture, desc) in enumerate(gestures_info):
            row = i // 3
            col = i % 3
        
            frame = tk.Frame(gesture_container, bg=background_color, bd=2, relief=tk.RIDGE)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
            tk.Label(frame, text=gesture, font=("Arial", 14, "bold"), 
                    bg=background_color, fg="white").pack(pady=5)
            tk.Label(frame, text=desc, font=("Arial", 12), 
                    bg=background_color, fg="white", wraplength=200).pack(pady=5)
        
            gesture_container.grid_columnconfigure(col, weight=1)
    
        gesture_container.grid_rowconfigure((len(gestures_info) // 3) + 1, weight=1)

    def show_about_page(self):
        self.main_content_frame.pack_forget()
        
        # Show about page and make it fill the space
        self.about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Ensure sidebar is visible
        self.sidebar_frame.place(x=0, y=0)

    def show_gestures_page(self):
        self.main_content_frame.pack_forget()
        self.about_frame.pack_forget()
        self.gestures_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.sidebar_frame.place(x=0, y=0)

    def show_main_content(self):
        self.about_frame.pack_forget()
        self.gestures_frame.pack_forget()
        self.main_content_frame.pack(fill=tk.BOTH, expand=True)
        self.sidebar_frame.place(x=0, y=0)

    def hide_main_content(self):
        """Hide all main content widgets"""
        self.preview_label.pack_forget()
        self.status_label.pack_forget()
        self.gesture_label.pack_forget()
        # Add any other widgets that need to be hidden

    def toggle_sidebar(self):
        x = self.sidebar_frame.winfo_x()
        if x < 0:
            self.sidebar_frame.place(x=0, y=0)
        else:
            self.sidebar_frame.place(x=-200, y=0)
        self.sidebar_icon.lift()
    
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
    
    def close_all_gui(self):
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        self.root.destroy()

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

def open_link(url):
    webbrowser.open_new(url)

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