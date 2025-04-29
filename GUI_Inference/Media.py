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
                fg="white", width=10, height=2
            )
            if option == "Home":
                btn.config(command=self.show_main_content)
            elif option == "Gestures":
                btn.config(command=self.show_gestures_page)
            elif option == "About":
                btn.config(command=self.show_about_page)
                
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

        # Load images for buttons
        script_dir = os.path.dirname(os.path.abspath(__file__))
        start_pause_img_path = os.path.join(script_dir, "../Icon/Video_Icon/start.png")
        stop_img_path = os.path.join(script_dir, "../Icon/Video_Icon/stop.png")
        exit_img_path = os.path.join(script_dir, "../Icon/Video_Icon/logout.png")

        self.start_pause_img = tk.PhotoImage(file=start_pause_img_path)
        self.stop_img = tk.PhotoImage(file=stop_img_path)
        self.exit_img = tk.PhotoImage(file=exit_img_path)

        # Replace buttons with images
        self.record_button = tk.Button(
            self.button_frame, 
            image=self.start_pause_img, 
            command=self.toggle_recording,
            bg=background_color,
            borderwidth=0
        )
        self.record_button.pack(side=tk.LEFT, padx=10)

        self.full_stop_button = tk.Button(
            self.button_frame, 
            image=self.stop_img, 
            command=self.full_stop_recording,
            bg=background_color,
            borderwidth=0,
            state=tk.DISABLED  # Initially disabled
        )
        self.full_stop_button.pack(side=tk.LEFT, padx=10)

        self.exit_button = tk.Button(
            self.button_frame, 
            image=self.exit_img, 
            command=self.close_app,
            bg=background_color,
            borderwidth=0
        )
        self.exit_button.pack(side=tk.LEFT)

        # About page frame (initially hidden)
        self.about_frame = tk.Frame(self.main_container, bg=background_color)
        tk.Label(self.about_frame, text="About Us", font=("Arial", 32), bg=background_color, fg="white").pack(pady=10)

        left_group = tk.Frame(self.about_frame, bg=background_color)
        left_group.pack(pady=10, padx=20, fill=tk.X)

        try:
            left_qr_image = Image.open("../Icon/QR/git.png")
            left_qr_photo = ImageTk.PhotoImage(left_qr_image)
            left_qr_label = tk.Label(left_group, image=left_qr_photo, bg=background_color)
            left_qr_label.image = left_qr_photo  # Keep a reference
            left_qr_label.pack()
        except:
            pass

        left_link = tk.Label(left_group, text="Visit the GitHub Repo", font=("Arial", 14, "underline"), fg="cyan", bg=background_color, cursor="hand2")
        left_link.pack(pady=(5, 0))
        left_link.bind("<Button-1>", lambda e: open_link("https://github.com/Belrayy/Hand_gesture"))

        # Frame for the right QR code and link
        right_group = tk.Frame(self.about_frame, bg=background_color)
        right_group.pack(pady=10, padx=20, fill=tk.X)

        try:
            right_qr_image = Image.open("../Icon/QR/Web.png")
            right_qr_photo = ImageTk.PhotoImage(right_qr_image)
            right_qr_label = tk.Label(right_group, image=right_qr_photo, bg=background_color)
            right_qr_label.image = right_qr_photo
            right_qr_label.pack()
        except:
            pass

        right_link = tk.Label(right_group, text="Visit Our Web Site", font=("Arial", 14, "underline"), fg="cyan", bg=background_color, cursor="hand2")
        right_link.pack(pady=(5, 0))
        right_link.bind("<Button-1>", lambda e: open_link("https://belrayy.github.io/Hand_gesture/Web_site/index.html"))

        tk.Label(self.about_frame, text="We're two passionate engineering students building Prometheus as our end-of-year project. \n Prometheus is more than just a project, it's our vision of intuitive, touchless control in action.", 
                font=("Arial", 12), bg=background_color, fg="white").pack(pady=10)

        # Gestures page frame (initially hidden)
        self.gestures_frame = tk.Frame(self.main_container, bg=background_color)
        tk.Label(self.gestures_frame, text="Gesture Guide", font=("Arial", 32), bg=background_color, fg="white").pack(pady=20)

        self.gestures_info = [
            ("Open Hand", "All fingers extended", "../Icon/Gestures/open_hand.jpg"),
            ("Fist", "All fingers closed", "../Icon/Gestures/fist.jpg"),
            ("Thumbs Up", "Only thumb extended", "../Icon/Gestures/thumb_up.jpg"),
            ("Peace Sign", "Index and middle fingers extended", "../Icon/Gestures/peace_sign.jpg"),
            ("Pointing", "Only index finger extended", "../Icon/Gestures/pointing.jpg"),
            ("Hang Loose", "Thumb and pinky extended", "../Icon/Gestures/call.jpg"),
            ("Number Four", "All fingers except thumb extended", "../Icon/Gestures/four.jpg"),
            ("Number Three", "Index, middle, and ring fingers extended", "../Icon/Gestures/three.jpg"),
            ("Two", "Index and pinky fingers extended", "../Icon/Gestures/two.jpg"),
            ("Spider-Man", "Thumb, index, and pinky fingers extended", "../Icon/Gestures/spiderman.jpg"),
            ("Rock-on", "Thumb, ring, and pinky fingers extended", "../Icon/Gestures/rock.jpg"),
            ("Okay", "Thumb and index finger touching", "../Icon/Gestures/ok.jpg"),
            ("Gun", "Index finger extended and thumb up", "../Icon/Gestures/gun.jpg")
        ]
        self.gesture_page_index = 0

        # Navigation frame
        nav_frame = tk.Frame(self.gestures_frame, bg=background_color)
        nav_frame.pack(pady=30)

        self.left_arrow = tk.Button(nav_frame, text="←", font=("Arial", 24), width=3, command=self.prev_gesture)
        self.left_arrow.grid(row=0, column=0, padx=10)

        # Gesture display area
        self.gesture_img_label = tk.Label(nav_frame, bg=background_color)
        self.gesture_img_label.grid(row=0, column=1, padx=10)

        self.gesture_desc_label = tk.Label(nav_frame, text="", font=("Arial", 16), bg=background_color, fg="white", wraplength=400, justify="center")
        self.gesture_desc_label.grid(row=1, column=1, pady=10)

        self.page_indicator = tk.Label(nav_frame, text="", font=("Arial", 14), bg=background_color, fg="white")
        self.page_indicator.grid(row=2, column=1, pady=5)

        self.right_arrow = tk.Button(nav_frame, text="→", font=("Arial", 24), width=3, command=self.next_gesture)
        self.right_arrow.grid(row=0, column=2, padx=10)

        self.update_gesture_page()

    def show_about_page(self):
        self.main_content_frame.pack_forget()
        self.gestures_frame.pack_forget()
        self.about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.sidebar_frame.place(x=0, y=0)

    def show_gestures_page(self):
        self.main_content_frame.pack_forget()
        self.about_frame.pack_forget()
        self.gestures_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.sidebar_frame.place(x=0, y=0)

    def update_gesture_page(self):
        gesture, desc, img_path = self.gestures_info[self.gesture_page_index]
    
        try:
            # Get absolute path by joining with script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_img_path = os.path.join(script_dir, img_path)
        
            img = Image.open(full_img_path)
            img = img.resize((200, 200), Image.LANCZOS)  # Updated from ANTIALIAS to LANCZOS
            photo = ImageTk.PhotoImage(img)
            self.gesture_img_label.config(image=photo, text="")
            self.gesture_img_label.image = photo
        except Exception as e:
            print(f"Error loading image: {e}")  # Debug output
            self.gesture_img_label.config(
                image="", 
                text=f"Image not found\n{img_path}", 
                font=("Arial", 12), 
                fg="red"
            )
            self.gesture_img_label.image = None

        self.gesture_desc_label.config(text=f"{gesture}\n\n{desc}")
        self.page_indicator.config(text=f"Gesture {self.gesture_page_index + 1} of {len(self.gestures_info)}")
    
        self.left_arrow.config(state=tk.DISABLED if self.gesture_page_index == 0 else tk.NORMAL)
        self.right_arrow.config(state=tk.DISABLED if self.gesture_page_index == len(self.gestures_info) - 1 else tk.NORMAL)

    def prev_gesture(self):
        if self.gesture_page_index > 0:
            self.gesture_page_index -= 1
            self.update_gesture_page()

    def next_gesture(self):
        if self.gesture_page_index < len(self.gestures_info) - 1:
            self.gesture_page_index += 1
            self.update_gesture_page()

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
            if not self.recording:  # Start recording if not already recording
                self.start_recording()
                self.status_label.config(text="Recording", fg="red")
                self.record_button.config(text="Pause")
                self.full_stop_button.config(state=tk.NORMAL)  # Enable full stop button
        elif gesture in ["Fist", "Call Me"]:
            self.gesture_label.config(fg="red")
            if self.recording and not self.paused:  # Pause recording if currently recording
                self.pause_recording()
                self.status_label.config(text="Paused", fg="orange")
                self.record_button.config(text="Resume")
            elif self.recording and self.paused:  # Resume recording if currently paused
                self.resume_recording()
                self.status_label.config(text="Recording", fg="red")
                self.record_button.config(text="Pause")
        elif gesture == "Thumbs Up":
            self.gesture_label.config(fg="blue")
        elif gesture == "Peace Sign":
            self.gesture_label.config(fg="purple")
        elif gesture == "Pointing":
            self.gesture_label.config(fg="orange")
        else:
            self.gesture_label.config(fg="black")

    def close_app(self):
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