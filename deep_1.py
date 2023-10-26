import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import time


# Example image paths (change these to actual image paths)
ok_example_path = "C:\\@Sumit\\Deep_learning_algorithm\\OK images"
ng_example_path = "C:\\@Sumit\\Deep_learning_algorithm\\Valid"

# Function to load and preprocess images from a folder
def load_and_preprocess_images(folder_path, label, image_size):
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (image_size, image_size))
            img = img / 255.0  # Normalize pixel values to [0, 1]
            
            images.append(img)
            labels.append(label)
    
    return images, labels

# Function to check for new images and retrain the model
def check_and_retrain_model():
    global model
    global test_data
    global test_labels
    
    global ok_folder_path
    global ng_folder_path
    global selected_class

    class_label = 0 if selected_class.get() == "OK" else 1

    # Load and preprocess the images
    if class_label == 0:
        ok_images, ok_labels = load_and_preprocess_images(ok_folder_path, label=0, image_size=image_size)
        ng_images = []
        ng_labels = []
    else:
        ok_images = []
        ok_labels = []
        ng_images, ng_labels = load_and_preprocess_images(ng_folder_path, label=1, image_size=image_size)

    # Combine "OK" and "NG" data
    all_images = np.array(ok_images + ng_images)
    all_labels = np.array(ok_labels + ng_labels)

    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    # Build the deep learning model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # keras.layersMaxPooling2D((2, 2))
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    messagebox.showinfo("Model Training", f"Test Accuracy: {test_accuracy}")

# Function to classify an image
def classify_image():
    if model is None:
        messagebox.showinfo("Model Not Trained", "Please train the model first.")
        return

    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(file_path)
        messagebox.showinfo("Image Classification", f"The image is classified as: {result}")

# Function to predict whether an image is OK or NG
def predict_image(image_path, image_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add a batch dimension

    # Make the prediction
    prediction = model.predict(img)

    # Check the result
    if prediction[0][0] < 0.5:
        return "OK"
    else:
        return "NG"

# Create the main application window
app = tk.Tk()
app.title("Image Classification App")

# Initialize variables
ok_folder_path = ""
ng_folder_path = ""
image_size = 224
selected_class = tk.StringVar()

# Create labels for paths
ok_path_label = tk.Label(app, text="OK Image Folder: " + ok_folder_path)
ng_path_label = tk.Label(app, text="NG Image Folder: " + ng_folder_path)

# Function to select OK and NG image folders
def select_ok_folder():
    global ok_folder_path
    ok_folder_path = filedialog.askdirectory()
    ok_path_label.config(text="OK Image Folder: " + ok_folder_path)
    check_and_retrain_model()  # Trigger training when a folder is selected

def select_ng_folder():
    global ng_folder_path
    ng_folder_path = filedialog.askdirectory()
    ng_path_label.config(text="NG Image Folder: " + ng_folder_path)
    check_and_retrain_model()  # Trigger training when a folder is selected

# Buttons to select OK and NG image folders
ok_folder_button = tk.Button(app, text="Select OK Image Folder", command=select_ok_folder)
ng_folder_button = tk.Button(app, text="Select NG Image Folder", command=select_ng_folder)
train_button = tk.Button(app, text="Train Model", command=check_and_retrain_model)
classify_button = tk.Button(app, text="Classify Image", command=classify_image)

# Radio buttons for selecting the class (OK or NG)
ok_radio = tk.Radiobutton(app, text="OK", variable=selected_class, value="OK")
ng_radio = tk.Radiobutton(app, text="NG", variable=selected_class, value="NG")

# Function to show an image in the GUI
def show_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img = ImageTk.PhotoImage(img)
    label = tk.Label(image=img)
    label.image = img
    label.pack()

# Create labels for images
ok_example_label = tk.Label(app, text="Example OK Image:")
ng_example_label = tk.Label(app, text="Example NG Image:")

# Example image paths (change these to actual image paths)
ok_example_path = "C:\\@Sumit\\Deep_learning_algorithm\\OK images"
ng_example_path = "C:\\@Sumit\\Deep_learning_algorithm\\Valid"

# Show example images
# show_image(ok_example_path)
# show_image(ng_example_path)

# Pack labels and buttons
ok_path_label.pack()
ng_path_label.pack()
ok_folder_button.pack()
ng_folder_button.pack()
ok_radio.pack()
ng_radio.pack()
train_button.pack()
classify_button.pack()
ok_example_label.pack()
ng_example_label.pack()

app.mainloop()
