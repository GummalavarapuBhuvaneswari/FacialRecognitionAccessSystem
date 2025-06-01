import cv2
import face_recognition
import numpy as np
import pandas as pd
import datetime
import os

# File to store user data
DATA_FILE = "authorized_users.csv"

# Step 1: Create CSV file if it doesn't exist
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=["Name", "Timestamp"] + [f"enc_{i}" for i in range(128)])
    df.to_csv(DATA_FILE, index=False)

# Step 2: Capture webcam
video = cv2.VideoCapture(0)
print("[INFO] Press 'S' to scan and register face, or 'Q' to quit.")

user_registered = False

while True:
    ret, frame = video.read()
    display_frame = frame.copy()

    # Display instruction on the screen
    cv2.putText(display_frame, "Press 'S' to Scan and Register | Press 'Q' to Quit",
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if user_registered:
        cv2.putText(display_frame, "User Registered Successfully!",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    cv2.imshow("Register - Facial Recognition", display_frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 0:
            print("[WARNING] No face detected. Try again.")
            continue

        encoding = face_encodings[0]
        name = input("Enter the name of the user: ")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = [name, timestamp] + list(encoding)
        df = pd.read_csv(DATA_FILE)
        df.loc[len(df)] = row
        df.to_csv(DATA_FILE, index=False)

        print(f"[SUCCESS] User '{name}' registered successfully!")
        user_registered = True

    elif key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
