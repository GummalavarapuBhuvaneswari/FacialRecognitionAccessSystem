import cv2
import face_recognition
import pandas as pd
import numpy as np
import datetime
import os
import subprocess

# --- Config / paths (moved to the top just so I don’t forget where they are)
USERS_FILE = "authorized_users.csv"
LOGS_FILE = "access_logs.csv"
REG_SCRIPT = "add_user.py"

# Quick sanity check — do we even have a list of users yet?
if not os.path.exists(USERS_FILE):
    print("[ERROR] User file missing. Run registration first maybe?")
    exit(1)

# Reading in the user data. This assumes there's a 'Name' and some encoded vectors
try:
    df_users = pd.read_csv(USERS_FILE)
except Exception as e:
    print("[ERROR] Couldn't read user file:", e)
    exit(1)

# Grab names and encoding data (excluding timestamp for now — maybe useful later?)
all_names = df_users["Name"].tolist()
known_encodings = df_users.drop(columns=["Name", "Timestamp"]).values.astype(float)

# If logs file doesn’t exist, create an empty one (kinda essential to track who came in)
if not os.path.isfile(LOGS_FILE):
    pd.DataFrame(columns=["Name", "Date-Time", "Status"]).to_csv(LOGS_FILE, index=False)

# Fire up the webcam — should probably add a fallback for external cams later
cam = cv2.VideoCapture(0)
print("[INFO] System armed. Press 'A' to register new face | Press 'Q' to quit.")

# main loop
while True:
    grabbed, frame = cam.read()
    if not grabbed:
        print("Webcam glitch — skipping frame.")
        continue

    # Converting from BGR to RGB (face_recognition prefers it that way)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting faces and getting encodings for them
    face_positions = face_recognition.face_locations(rgb_frame)
    frame_encodings = face_recognition.face_encodings(rgb_frame, face_positions)

    for (top, right, bottom, left), encoding in zip(face_positions, frame_encodings):
        label = "Unknown"
        status = "Access Denied"
        color = (0, 0, 255)  # default: red box

        # measure similarity against stored encodings
        distances = face_recognition.face_distance(known_encodings, encoding)
        closest_index = np.argmin(distances) if len(distances) > 0 else -1

        # Just a rough cutoff; might tweak this based on results
        if closest_index >= 0 and distances[closest_index] < 0.45:
            label = all_names[closest_index]
            status = "Access Granted"
            color = (0, 255, 0)  # yay, green box

        # Draw box and text on screen
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{label} - {status}", (left, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Optional: flash status at the top for better visibility
        cv2.putText(frame, f"{status}: {label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        # Logging time & decision
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            logs = pd.read_csv(LOGS_FILE)
        except:
            logs = pd.DataFrame(columns=["Name", "Date-Time", "Status"])

        new_log = pd.DataFrame([[label, timestamp, status]], columns=logs.columns)
        logs = pd.concat([logs, new_log], ignore_index=True)
        logs.to_csv(LOGS_FILE, index=False)

    # footer instructions
    footer_text = "Press 'A' to Register | Press 'Q' to Quit"
    cv2.putText(frame, footer_text, (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display updated frame
    cv2.imshow("Access Control", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        print("[INFO] Shutting down system.")
        break
    elif key == ord('a'):
        print("[INFO] Registering a new user (launching external script).")
        cam.release()
        cv2.destroyAllWindows()
        subprocess.run(["python", REG_SCRIPT])
        break

# cleanup — never forget this or the webcam locks up
cam.release()
cv2.destroyAllWindows()
