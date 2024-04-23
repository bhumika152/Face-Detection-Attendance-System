from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

# Load face recognition data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Create 'facetracker' folder if it doesn't exist
if not os.path.exists('face_tracker'):
    os.makedirs('face_tracker')

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Initialize background image
imgBackground = cv2.imread("background2.png")

# Check if the background image was loaded successfully
if imgBackground is None:
    print("Error: Could not load the background image.")
else:
    # Define column names for the attendance CSV
    COL_NAMES = ['NAME', 'ENTER_TIME', 'EXIT_TIME']

    # Initialize variables
    prev_names = []
    current_date = None
    entry_time_dict = {}

    while True:
        ret, frame = video.read()

        # Check for end of video or if frame is empty
        if not ret or frame is None:
            print("Error: Couldn't read frame from the video source.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        names = []
        for (x, y, w, h) in faces:
            # Extract the face region
            crop_img = frame[y:y + h, x:x + w, :]

            # Resize and flatten the face image for recognition
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

            # Predict the label (person) using KNN
            output = knn.predict(resized_img)
            names.append(str(output[0]))

            # Get timestamp for attendance
            timestamp = datetime.now().strftime("%H:%M:%S")
            date = datetime.now().strftime("%d-%m-%Y")

            # Display rectangle and information on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 255, 50), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 255, 50), -1)
            cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Check for entry or exit
        entered_names = list(set(names) - set(prev_names))
        exited_names = list(set(prev_names) - set(names))

        for name in entered_names:
            if name not in entry_time_dict:
                entry_time_dict[name] = timestamp

        for name in exited_names:
            if name in entry_time_dict:
                entry_time = entry_time_dict.pop(name)
                with open(f"face_tracker/{date}_attendance.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if os.path.getsize(f"face_tracker/{date}_attendance.csv") == 0:
                        writer.writerow(COL_NAMES)
                    writer.writerow([name, entry_time, timestamp])

        prev_names = names

        # Update the background image with the current frame
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Frame", imgBackground)

        # Check for keypress events
        k = cv2.waitKey(1)
        if k == ord('x'):
            break

# Release video capture object and close windows
video.release()
cv2.destroyAllWindows()
