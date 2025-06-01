import cv2
import face_recognition
import numpy as np
import pandas as pd
import datetime
import os

# File to store user facial encodings and metadata
CSV_PATH = "authorized_users.csv"

# Make sure the CSV exists before we do anything else
if not os.path.isfile(CSV_PATH):
    # Building column headers manually (might be a cleaner way, but this is simple)
    header_cols = ["Name", "Timestamp"]
    for idx in range(128):
        header_cols.append(f"enc_{idx}")
    
    # Create empty DataFrame with correct columns
    df_init = pd.DataFrame(columns=header_cols)
    df_init.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Created new user data file at {CSV_PATH}")

# Fire up the webcam
cam = cv2.VideoCapture(0)
print("[INFO] Facial registration started. Hit 'S' to scan, or 'Q' to quit.")

# Just tracking whether we've added a face during this run
user_added = False

while True:
    success, frame = cam.read()

    if not success:
        print("[ERROR] Camera read failed. Not sure why. Try restarting maybe?")
        continue

    # We'll work on a copy so we can draw text/boxes
    overlay = frame.copy()

    # Display instructions at the bottom of the screen
    cv2.putText(overlay, "Press 'S' to Scan and Register | Press 'Q' to Quit", 
                (20, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # After someone registers, show confirmation message
    if user_added:
        cv2.putText(overlay, "Registration Complete!", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 3)

    cv2.imshow("Register - Facial Recognition", overlay)

    # Check for keypress
    key = cv2.waitKey(1)

    if key == ord('s') or key == ord('S'):
        # Convert to RGB — face_recognition expects that
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Locate faces and get encodings
        locations = face_recognition.face_locations(rgb)
        found_encodings = face_recognition.face_encodings(rgb, locations)

        if not found_encodings:
            print("[WARNING] No face detected. Maybe adjust lighting or remove mask?")
            continue

        # We just grab the first face; not handling multiple faces (yet?)
        encoding = found_encodings[0]

        # Prompt for user's name
        name = input("Enter name for this person: ").strip()
        if name == "":
            print("[INFO] Blank name? Let's try that again.")
            continue

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Combine all the info into a row
        entry = [name, timestamp] + list(encoding)

        # Append to the CSV
        try:
            current_df = pd.read_csv(CSV_PATH)
            current_df.loc[len(current_df)] = entry
            current_df.to_csv(CSV_PATH, index=False)
            print(f"[SUCCESS] Added {name} to the database.")
            user_added = True
        except Exception as e:
            print(f"[ERROR] Something went wrong while saving: {e}")

    elif key == ord('q') or key == ord('Q'):
        print("[INFO] Exiting the registration script.")
        break

# Clean up the resources
cam.release()
cv2.destroyAllWindows()
