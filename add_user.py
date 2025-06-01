import cv2
import face_recognition
import numpy as np
import pandas as pd
import datetime
import os

# Where we're going to keep track of people we know
CSV_PATH = "authorized_users.csv"

# If the file isn't already there, we make it with headers for name, timestamp, and 128 facial encodings
if not os.path.isfile(CSV_PATH):
    # Added enc_0 to enc_127 just so it's easier to keep track of encoding values in CSV
    columns = ["Name", "Timestamp"]
    for i in range(128):
        columns.append(f"enc_{i}")
    starter_df = pd.DataFrame(columns=columns)
    starter_df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Created new data file at {CSV_PATH}")

# Initialize webcam feed (default camera)
cam = cv2.VideoCapture(0)

print("[INFO] Press 'S' to scan & register a face. Hit 'Q' when you're done.")

# A simple flag to check if we already registered someone in this run
has_registered = False

while True:
    grabbed, frame = cam.read()
    
    if not grabbed:
        print("[ERROR] Couldn't grab frame from camera.")
        continue

    # Just duplicating for overlay stuff
    screen = frame.copy()

    # Just reminding the user what to do
    cv2.putText(screen, "Press 'S' to Scan and Register | Press 'Q' to Quit", 
                (15, screen.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if has_registered:
        # This only shows up after a successful registration
        cv2.putText(screen, "User Registered Successfully!", 
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 3)

    # Show the video with text overlay
    cv2.imshow("Register - Facial Recognition", screen)

    # Grab key input
    key = cv2.waitKey(1)

    if key == ord('s') or key == ord('S'):
        # Convert color format since face_recognition works with RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Try finding any face(s) in the current frame
        faces = face_recognition.face_locations(rgb_image)
        encodings = face_recognition.face_encodings(rgb_image, faces)

        if not encodings:
            print("[WARNING] Hmm... no face detected. Try holding still or adjusting lighting.")
            continue

        face_encoding = encodings[0]  # Just taking the first face for now

        # Ask the user who this is
        user_name = input("Enter the name of the user: ").strip()

        # Getting current time, just for logging purposes
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Stitch everything together into a row
        full_data = [user_name, time_now] + list(face_encoding)

        # Read existing data and append
        existing_df = pd.read_csv(CSV_PATH)
        existing_df.loc[len(existing_df)] = full_data
        existing_df.to_csv(CSV_PATH, index=False)

        print(f"[SUCCESS] {user_name} has been added to the authorized list.")
        has_registered = True

    elif key == ord('q') or key == ord('Q'):
        print("[INFO] Quitting the registration system.")
        break

# Release the cam and close the OpenCV windows
cam.release()
cv2.destroyAllWindows()
