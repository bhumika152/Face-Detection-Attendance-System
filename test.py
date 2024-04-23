from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load face recognition data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

#background_path = "C:/Users/HP/OneDrive/Desktop/project1final/background.png"
#print("File path:", background_path)
#imgBackground = cv2.imread(background_path)
imgBackground=cv2.imread("background.png")


# Check if the background image was loaded successfully
if imgBackground is None:
    print(f"Error: Could not load the background image.")
else:
    
    # Define column names for the attendance CSV
    COL_NAMES = ['NAME', 'TIME']

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
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

            # Display rectangle and information on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 255, 50), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 255, 50), -1)
            cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # Record attendance information
            attendance = [str(output[0]), str(timestamp)]

        # Update the background image with the current frame
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Frame", imgBackground)

        # Check for keypress events
        k = cv2.waitKey(1)
        if k == ord('p'):
            # Check if the attendance file exists
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
            # Write attendance to CSV file
            with open("Attendance/Attendance_" + date + ".csv", "+a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    for i in names:  
                        speak("Attendance Taken"+i)
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)
                if exist:
                    speak("Attendance already taken")

        if k == ord('x'):
            break

# Release video capture object and close windows
video.release()
cv2.destroyAllWindows()



# This modification ensures that the background image is loaded successfully, and it adds the `newline=''` parameter to the CSV file handling to avoid extra blank lines in the output file.