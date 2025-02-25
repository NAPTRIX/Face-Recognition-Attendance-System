import cv2
import numpy as np
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk

# Define the paths
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_recognizer_path = 'face_recognizer.yml'
sign_up_images_dir = 'your_beautiful_images'
log_file = 'log_book.txt'

# Create the necessary directories if they don't exist
if not os.path.exists(sign_up_images_dir):
    os.makedirs(sign_up_images_dir)

# Initialize the face recognizer and the face cascade
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Function to collect images for sign-up
def sign_up(name):
    cap = cv2.VideoCapture(0)
    count = 0

    while count < 30:  # The more, the better the recognition accuracy
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            cv2.imwrite(f'{sign_up_images_dir}/{name}_{count}.jpg', face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Face {count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Sign Up', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# train the face recognizer
def train_recognizer():
    images, labels = [], []
    label_dict = {}
    label_id = 0

    for image_name in os.listdir(sign_up_images_dir):
        image_path = os.path.join(sign_up_images_dir, image_name)
        label_name = image_name.split('_')[0]

        if label_name not in label_dict:
            label_dict[label_name] = label_id
            label_id += 1

        images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        labels.append(label_dict[label_name])

    if images and labels:
        face_recognizer.train(images, np.array(labels))
        face_recognizer.save(face_recognizer_path)

        with open('labels.txt', 'w') as f:
            for name, label in label_dict.items():
                f.write(f'{name}:{label}\n')
    else:
        messagebox.showerror("Error", "No images found to train the recognizer.")

# Function to sign in and log detected faces
def sign_in():
    if not os.path.exists(face_recognizer_path) or not os.path.exists('labels.txt'):
        messagebox.showerror("Error", "No trained data found. Please sign up first.")
        return

    try:
        face_recognizer.read(face_recognizer_path)
    except cv2.error as e:
        messagebox.showerror("Error", f"Error loading face recognizer: {e}")
        return

    label_dict = {}
    with open('labels.txt', 'r') as f:
        for line in f.readlines():
            name, label = line.strip().split(':')
            label_dict[int(label)] = name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    current_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        new_names = set()

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            try:
                label, confidence = face_recognizer.predict(face)
                if label in label_dict and confidence < 100:
                    name = label_dict[label]
                    new_names.add(name)
                    if name not in current_names:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(log_file, 'a') as log:
                            log.write(f'{timestamp} - {name} entered\n')
                        current_names.add(name)
                    cv2.putText(frame, f'{name} - {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'INTRUDA IN DA HOUSE!!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except cv2.error as e:
                print(f"Error predicting face: {e}")
                cv2.putText(frame, 'Prediction Error', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Log when people leave the frame
        for name in current_names - new_names:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(log_file, 'a') as log:
                log.write(f'{timestamp} - {name} left\n')
        current_names = new_names

        cv2.imshow('Sign In', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if not os.path.exists(face_recognizer_path) or not os.path.exists('labels.txt'):
        messagebox.showerror("Error", "No trained data found. Please sign up first.")
        return

    face_recognizer.read(face_recognizer_path)
    label_dict = {}

    with open('labels.txt', 'r') as f:
        for line in f.readlines():
            name, label = line.strip().split(':')
            label_dict[int(label)] = name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    current_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        new_names = set()

        for (x, y, w, h) in faces:
            face = gray[y + h, x + w]
            try:
                label, confidence = face_recognizer.predict(face)
                if label in label_dict and confidence < 100:
                    name = label_dict[label]
                    new_names.add(name)
                    if name not in current_names:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(log_file, 'a') as log:
                            log.write(f'{timestamp} - {name} entered\n')
                        current_names.add(name)
                    cv2.putText(frame, f'{name} - {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'INTRUDA IN DA HOUSE!!', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except cv2.error:
                messagebox.showerror("Error", "Could not predict face. Please sign up first.")
                return

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Log when people leave the frame
        for name in current_names - new_names:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(log_file, 'a') as log:
                log.write(f'{timestamp} - {name} left\n')
        current_names = new_names

        cv2.imshow('Sign In', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Delete a face
def delete_face(name):
    for image_name in os.listdir(sign_up_images_dir):
        if image_name.startswith(name):
            os.remove(os.path.join(sign_up_images_dir, image_name))

    if os.path.exists('labels.txt'):
        with open('labels.txt', 'r') as f:
            lines = f.readlines()

        with open('labels.txt', 'w') as f:
            for line in lines:
                if not line.startswith(name + ':'):
                    f.write(line)

    train_recognizer()
    messagebox.showinfo("Deleted", f'Face data for {name} has been deleted.')

# Main GUI function
def main_gui():
    root = tk.Tk()
    root.title("Face Recognition Attendance System")

    # set window size and background
    root.geometry("700x500")
    bg_image = Image.open("background.png")  # Ensure you have a background.png or whatever format you'd like in the same directory
    bg_image = bg_image.resize((700, 500), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)

    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)

    # Title label
    title_label = tk.Label(root, text="Face Recognition Attendance System", font=("Helvetica", 16, "bold"), bg="#ADD8E6")
    title_label.pack(pady=20)

    # buttons
    def handle_signup():
        name = simpledialog.askstring("Sign Up", "Enter your name:")
        if name:
            sign_up(name)
            train_recognizer()

    def handle_signin():
        sign_in()

    def handle_delete():
        name = simpledialog.askstring("Delete Face", "Enter the name to delete:")
        if name:
            delete_face(name)

    sign_up_button = tk.Button(root, text="Sign Up", font=("Helvetica", 14), command=handle_signup, bg="#4CAF50", fg="white", padx=20, pady=10)
    sign_in_button = tk.Button(root, text="Sign In", font=("Helvetica", 14), command=handle_signin, bg="#008CBA", fg="white", padx=20, pady=10)
    delete_button = tk.Button(root, text="Delete Face", font=("Helvetica", 14), command=handle_delete, bg="#f44336", fg="white", padx=20, pady=10)

    sign_up_button.pack(pady=10)
    sign_in_button.pack(pady=10)
    delete_button.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    main_gui()
