import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Paths to the YOLO files
yolo_cfg = "C:/Users/mogil/shreyas/yolo/yolov3.cfg"
yolo_weights = "C:/Users/mogil/shreyas/yolo/yolov3.weights"
coco_names = "C:/Users/mogil/shreyas/yolo/coco.names"

# Load YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# Enable GPU if available
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Read COCO class names
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        self.root.geometry("1920x1080")  # Set to screen resolution if needed
        self.root.state('zoomed')  # Maximize window

        # Fonts and colors
        self.title_font = ("Helvetica", 16, "bold")
        self.button_font = ("Helvetica", 12)
        self.status_font = ("Helvetica", 20)
        self.bg_color = "#f0f0f0"
        self.button_bg = "#4CAF50"
        self.button_fg = "white"

        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.button_frame.pack(pady=5)

        self.start_button = tk.Button(self.button_frame, text="Start", font=self.button_font, bg=self.button_bg, fg=self.button_fg, command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = tk.Button(self.button_frame, text="Stop", font=self.button_font, bg=self.button_bg, fg=self.button_fg, command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10)

        self.status_bar = tk.Label(self.main_frame, text="Status: Ready", font=self.status_font, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.object_count_label = tk.Label(self.main_frame, text="", font=self.status_font, bg=self.bg_color)
        self.object_count_label.pack()

        self.cap = None
        self.is_detecting = False

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.update_status("Error: Could not open camera.")
            return

        # Get screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("Status: Detecting objects...")

        self.is_detecting = True
        self.detect_objects()

    def detect_objects(self):
        ret, frame = self.cap.read()
        if ret:
            # Prepare the frame for detection
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Process the detections
            class_ids = []
            confidences = []
            boxes = []
            confidence_threshold = 0.5
            nms_threshold = 0.4

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

            # Draw bounding boxes and labels on the frame
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Count the number of each detected object
            object_counts = {}
            for i in indexes:
                label = classes[class_ids[i]]
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1

            # Update object count label
            count_text = "  ".join([f"{label}: {count}" for label, count in object_counts.items()])
            self.object_count_label.config(text=count_text)

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to ImageTk format
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            # Update the label with the new image
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        if self.is_detecting:
            self.root.after(10, self.detect_objects)
        else:
            self.cap.release()
            self.video_label.config(image=None)
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.update_status("Status: Ready")

    def stop_detection(self):
        self.is_detecting = False

    def update_status(self, message):
        self.status_bar.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
