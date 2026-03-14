import cv2
import time
import subprocess
from posture_detector import PostureDetector

def main():
    # Attempt to open the default camera (index 0)
    # If using an external webcam, this might be 1 or higher.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize our Posture Detector Class
    detector = PostureDetector()

    # Variables for FPS calculation
    pTime = 0
    audio_process = None

    print("Starting webcam stream... Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam. Exiting...")
            break

        # Flip image horizontally for a mirror effect (more intuitive for users)
        img = cv2.flip(img, 1)

        # Output the image through the pose detector
        img = detector.find_pose(img, draw=True)
        
        # Extract the landmarks from the image
        landmarks = detector.get_landmarks(img)
        
        # Analyze the posture
        status_text, color = detector.check_posture(landmarks)
        
        # Audio Alert Logic
        # (0, 0, 255) is the BGR color for Red
        if color == (0, 0, 255):
            # If the audio isn't currently playing, start it
            if audio_process is None or audio_process.poll() is not None:
                try:
                    # using native macOS afplay to play the mp3 in the background
                    audio_process = subprocess.Popen(["afplay", "audio.mp3"])
                except Exception as e:
                    print("Ensure 'audio.mp3' is present.", e)
        else:
            # If posture is corrected and sound is playing, terminate it
            if audio_process is not None and audio_process.poll() is None:
                audio_process.terminate()
                audio_process = None

        # Display FPS
        # Calculate Frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        # Draw Information on screen
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Show the processed frame
        cv2.imshow("Posture Corrector", img)

        # Wait for 1 millisecond and check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if audio_process is not None and audio_process.poll() is None:
        audio_process.terminate()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
