
import cv2
import face_recognition
import pandas as pd
import numpy as np
import datetime
import os
import subprocess

# File paths
DATA_FILE = "authorized_users.csv"
LOG_FILE = "access_logs.csv"
ADD_USER_SCRIPT = "add_user.py"

# Step 1: Load user encodings
if not os.path.exists(DATA_FILE):
    print("[ERROR] authorized_users.csv not found. Run add_user.py first.")
    exit()

df = pd.read_csv(DATA_FILE)

known_names = df["Name"].tolist()
known_encodings = df.drop(columns=["Name", "Timestamp"]).values.astype(float)

# Step 2: Prepare log file
if not os.path.exists(LOG_FILE):
    log_df = pd.DataFrame(columns=["Name", "Date-Time", "Status"])
    log_df.to_csv(LOG_FILE, index=False)

# Step 3: Start webcam
video = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit, or 'a' to add new user.")

while True:
    ret, frame = video.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        status = "Access Denied"
        color = (0, 0, 255)

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Adjusted threshold for better accuracy
        if face_distances[best_match_index] < 0.45:
            name = known_names[best_match_index]
            status = "Access Granted"
            color = (0, 255, 0)

        # Draw rectangle and label near face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} - {status}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw center status message
        cv2.putText(frame, f"{status}: {name}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Log the attempt
        log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_df = pd.read_csv(LOG_FILE)
        log_df.loc[len(log_df)] = [name, log_time, status]
        log_df.to_csv(LOG_FILE, index=False)

    # Display prompt message at the bottom
    cv2.putText(frame, "Press 'A' to Register | Press 'Q' to Quit", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Access Control", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('a'):
        print("[INFO] You pressed 'a'. Opening face registration...")
        video.release()
        cv2.destroyAllWindows()
        subprocess.run(["python", ADD_USER_SCRIPT])
        break

video.release()
cv2.destroyAllWindows()
