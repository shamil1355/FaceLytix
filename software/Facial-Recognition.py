import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition

# Paths
image_path = r'D:\EKC\MINI PROJECT\Attandence system\image_folder'
attendance_folder = os.path.join(os.getcwd(), 'attendance')
csv_path = os.path.join(attendance_folder, 'Attendance.csv')
camera_url = 'http://192.168.210.21/cam-hi.jpg'

# Ensure Attendance Folder Exists
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

# Ensure CSV File Exists with Proper Headers
if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(csv_path, index=False)

# Load Images & Encode Faces
if not os.path.exists(image_path):
    print(f"Error: Image folder '{image_path}' does not exist!")
    exit()

images = []
classNames = []
file_list = os.listdir(image_path)
print("Images Found:", file_list)

for file in file_list:
    curImg = cv2.imread(os.path.join(image_path, file))
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(file)[0])

print("Classes:", classNames)


# Function to Encode Faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:  # Only append if encoding exists
            encodeList.append(encodes[0])
    return encodeList


# Function to Mark Attendance
def markAttendance(name):
    df = pd.read_csv(csv_path)

    # Check for missing headers & fix
    if "Name" not in df.columns or "Time" not in df.columns:
        print("⚠ CSV file corrupted or missing headers. Resetting...")
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(csv_path, index=False)

    # Prevent duplicate marking
    if name not in df["Name"].values:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        new_entry = pd.DataFrame([[name, dtString]], columns=["Name", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(csv_path, index=False)


# Encode Faces
encodeListKnown = findEncodings(images)
print('✅ Encoding Complete')

# Face Recognition Loop
while True:
    try:
        img_resp = urllib.request.urlopen(camera_url, timeout=5)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
    except Exception as e:
        print(f"⚠ Error accessing camera: {e}")
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.4)  # Stricter threshold
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) > 0:
            best_match_index = np.argmin(faceDis)
            if matches[best_match_index] and faceDis[best_match_index] < 0.4:  # Ensuring stricter match
                name = classNames[best_match_index].upper()
            else:
                name = "Unknown"
        else:
            name = "Unknown"

        # Draw on Image
        y1, x2, y2, x1 = [v * 4 for v in faceLoc]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Mark attendance only for known faces
        if name != "Unknown":
            markAttendance(name)

    cv2.imshow('ESP32 Camera Feed', img)
    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
