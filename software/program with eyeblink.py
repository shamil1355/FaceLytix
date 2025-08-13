import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition
from scipy.spatial import distance
import dlib

# Paths
image_path = r'D:\EKC\MINI PROJECT\Attandence system\image_folder'
attendance_folder = os.path.join(os.getcwd(), 'attendance')
csv_path = os.path.join(attendance_folder, 'Attendance.csv')
camera_url = 'http://192.168.137.239/cam-hi.jpg'

# Ensure Attendance Folder Exists
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

# Ensure CSV File Exists with Proper Headers
if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
    df = pd.DataFrame(columns=["Name", "Time", "Verification"])
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

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(face_landmarks):
    # Get coordinates for both eyes
    left_eye = face_landmarks['left_eye']
    right_eye = face_landmarks['right_eye']
    
    # Calculate eye aspect ratio for both eyes
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    
    # Average the eye aspect ratio
    ear = (left_ear + right_ear) / 2.0
    return ear < 0.2  # Threshold for blink detection

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
def markAttendance(name, verification_status):
    df = pd.read_csv(csv_path)

    # Check for missing headers & fix
    if "Name" not in df.columns or "Time" not in df.columns or "Verification" not in df.columns:
        print("⚠ CSV file corrupted or missing headers. Resetting...")
        df = pd.DataFrame(columns=["Name", "Time", "Verification"])
        df.to_csv(csv_path, index=False)

    # Get current date
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    
    # Check if person has already been marked for today
    today_entries = df[df["Time"].str.contains(current_date, na=False)]
    if name not in today_entries["Name"].values:
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        new_entry = pd.DataFrame([[name, dtString, verification_status]], 
                               columns=["Name", "Time", "Verification"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"✅ Attendance marked for {name} - {verification_status}")
    else:
        print(f"⚠ {name} already marked attendance today")

# Encode Faces
encodeListKnown = findEncodings(images)
print('✅ Encoding Complete')

# Variables for blink detection
blink_counter = {}
REQUIRED_BLINKS = 2
BLINK_TIME_WINDOW = 5  # seconds

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
    face_landmarks_list = face_recognition.face_landmarks(imgS, facesCurFrame)

    for encodeFace, faceLoc, face_landmarks in zip(encodesCurFrame, facesCurFrame, face_landmarks_list):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.4)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        if len(faceDis) > 0:
            best_match_index = np.argmin(faceDis)
            if matches[best_match_index] and faceDis[best_match_index] < 0.4:
                name = classNames[best_match_index].upper()
                current_time = datetime.now()

                # Initialize blink counter for new faces
                if name not in blink_counter:
                    blink_counter[name] = {
                        'count': 0,
                        'start_time': current_time,
                        'verified': False
                    }

                # Check for blink
                if detect_blink(face_landmarks):
                    blink_counter[name]['count'] += 1
                    
                # Check if enough blinks within time window
                time_diff = (current_time - blink_counter[name]['start_time']).total_seconds()
