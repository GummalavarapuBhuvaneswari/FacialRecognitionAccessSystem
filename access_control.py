import cv2
import face_recognition
import pandas as pd
import numpy as np
import datetime
import os
import subprocess

# File locations – just keeping things centralized up here
USERS_FILE = "authorized_users.csv"
LOGS_FILE = "access_logs.csv"
REG_SCRIPT = "add_user.py"

# First thing: make sure we actually have registered users to compare with
if not os.path.exists(USERS_FILE):
    print("[ERROR] Couldn't find the authorized users file. Try running add_user.py first.")
    exit(1)

# Load known face encodings from CSV
user_df = pd.read_csv(USERS_FILE)

# These will be used later to compare faces
names_known = user_df["Name"].tolist()
face_data = user_df.drop(columns=["Name", "Timestamp"]).values.astype(float)

# Make sure we’ve got a place to log entry attempts
if not os.path.isfile(LOGS_FILE):
    logs_df = pd.DataFrame(columns=["Name", "Date-Time", "Status"])
    logs_df.to_csv(LOGS_FILE, index=False)

# Open up webcam feed
cam = cv2.VideoCapture(0)
print("[INFO] Facial recognition is running. Press 'A' to add user or 'Q' to quit.")

while True:
    ok, frame = cam.read()
    if not ok:
        print("[ERROR] Couldn't read from the webcam.")
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Try to find faces in current frame
    found_faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, found_faces)

    for (top, right, bottom, left), this_encoding in zip(found_faces, encodings):
        match_name = "Unknown"
        access_status = "Access Denied"
        box_color = (0, 0, 255)  # Red by default

        # Figure out the closest face from known list
        distances = face_recognition.face_distance(face_data, this_encoding)
        best_idx = np.argmin(distances)

        # Check if the best match is close enough – kinda arbitrary but tuned a bit
        if distances[best_idx] < 0.45:
            match_name = names_known[best_idx]
            access_status = "Access Granted"
            box_color = (0, 255, 0)  # Green if it's a match

        # Draw a box around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.putText(frame, f"{match_name} - {access_status}", 
                    (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # Show access result in the center too
        cv2.putText(frame, f"{access_status}: {match_name}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

        # Record the attempt in logs
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        old_logs = pd.read_csv(LOGS_FILE)
        new_row = [match_name, now_str, access_status]
        old_logs.loc[len(old_logs)] = new_row
        old_logs.to_csv(LOGS_FILE, index=False)

    # Add bottom message to guide user
    cv2.putText(frame, "Press 'A' to Register | Press 'Q' to Quit", 
                (15, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show the camera feed with overlays
    cv2.imshow("Access Control", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        print("[INFO] Exiting the system.")
        break
    elif key == ord('a'):
        print("[INFO] Launching user registration script...")
        cam.release()
        cv2.destroyAllWindows()
        subprocess.run(["python", REG_SCRIPT])
        break

# Wrap it up
cam.release()
cv2.destroyAllWindows()
