# Posture Corrector AI

An OpenCV and MediaPipe-based machine-learning model that tracks your posture through your webcam. If the model detects that you're slouching, it automatically alerts you on-screen and plays an audio warning.

## How it Works
The application leverages **Google's MediaPipe** to natively recognize your body without the need for custom training data:

1. **Posture / Slouching (`pose_landmarker_heavy.task`)**: We extract the coordinates for your Ear, Shoulder, and Hip. Using trigonometric calculations, the application monitors your neck inclination and torso inclination.
2. **Drowsiness / Shut Eyes (`face_landmarker.task`)**: The model detects 478 3D facial landmarks. We calculate the Eye Aspect Ratio (EAR) of your eyelids to ensure they are open.
3. **Leaning / Hand Support (`hand_landmarker.task`)**: The model tracks 21 hand-knuckle coordinates. If your knuckles intersect with the radius of your chin, it warns you that you are leaning on your hands.

If you hunch forward past the configured comfort threshold, shut your eyes, or lean on your hands, the application warns you and triggers a native MacOS `afplay` subprocess to sound a warning tone (`audio.mp3`).

## Project Structure

### `main.py`
The entry point of the application. It handles the core loop that talks to your webcam, processes the image, and paints the feedback onto your screen.
- **`main()`**: Initializes `cv2.VideoCapture`, reads frames infinitely, sends them to the posture detector, and plays/stops the background audio process based on the posture state.

### `posture_detector.py`
The class handling the machine learning integration and posture logic.
- **`PostureDetector(Class)`**:
  - `__init__()`: Loads the MediaPipe vision landmarker utilizing the `pose_landmarker_heavy.task` machine learning weights.
  - `find_pose(img, draw=True)`: Feeds the numpy image matrix into the MediaPipe Task API, then loops over the detected body parts and draws OpenCV circles on the recognized landmarks (Ear, Shoulder, Hip, etc.).
  - `get_landmarks(img)`: Extracts the X and Y coordinates (relative to image width/height) of the user's body landmarks into an array of dictionaries.
  - `check_posture(landmarks)`: The core logic. Extracts the right Ear, Shoulder, and Hip coordinates. Computes the deviation angle from a straight vertical line. If the absolute neck inclination is **> 35 degrees**, it returns a "Slouching" state.

### `utils.py`
Helper math functions utilized by the module.
- **`calculate_angle(a, b, c)`**: Uses `numpy.arctan2` to figure out the geometric angle between three (x, y) coordinates where `b` is the middle vertex.
- **`calculate_distance(point1, point2)`**: Uses `math.hypot` to calculate Euclidean distance between two coordinates if needed for alternative tracking approaches.

## Setup & Installation

This project uses a python **virtual environment (`venv`)** to safely sandbox its dependencies (`opencv-python`, `mediapipe`, `numpy`) away from your global MacOS system python.

1. Create the virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
   *(You must run this command in your terminal each time you open a new window to work on the project).*
3. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
4. Verify you have an `audio.mp3` file in the same directory as the script.
5. Run the application!
   ```bash
   python main.py
   ```
   *Press 'q' on your keyboard to quit the application.*

---

# 🚀 Future Updates: Hardware & Software Integration

This document outlines the roadmap for integrating the **Software (Neck Tracker)** with the **Hardware (Back Tracker)** to create a unified, real-time posture and ergonomics coach.

## 1. Hardware-to-Software Communication Bridge
Currently, the neck tracking runs independently on the computer using Python and a webcam. The next phase will bring the ESP32 back-tracking data into the same Python environment.
- **Communication Protocol:** Bluetooth Low Energy (BLE) or Serial Communication (USB).
- **Data Format:** The ESP32 will send lightweight state updates (e.g., `{"back_posture": "good"}` or `{"back_posture": "slouch", "pitch": 45}`).
- **Python Integration:** We will use the `pyserial` library (for USB) or `bleak` (for BLE). This will run on a separate background thread in Python so it doesn't block the real-time OpenCV camera feed.

## 2. Python as the "Master Node"
The Python script will be upgraded from a simple OpenCV display into the brain of the entire operation, maintaining the global state of the user's overall posture.
- **Unified State Evaluation:** The script will evaluate `is_neck_good` (from MediaPipe) AND `is_back_good` (from the ESP32).
- **Combined Logic:**
  ```python
  if is_neck_good and is_back_good:
      overall_status = "Perfect"
  elif not is_neck_good and is_back_good:
      overall_status = "Tech Neck Detected"
  elif is_neck_good and not is_back_good:
      overall_status = "Back Slouching"
  else:
      overall_status = "Poor Overall Posture"
  ```

## 3. Two-Way Feedback System
- **Unified Visual Feedback:** The OpenCV video window will display both the neck angle and the live back status received from the ESP32.
- **Unified Haptic Feedback:** If the Python script detects prolonged "Tech Neck" with the camera (even if the back is straight), Python will send a command *back* to the ESP32 to trigger the vibration motor. This effectively turns the ESP32 into a universal alert buzzer for the whole system.

## 4. Gamification & The 25-Minute Focus Session
- **Smart Timer:** We will build a 25-minute Pomodoro timer in the Python UI.
- **Productivity Points:** The timer/score only increments while `overall_status == "Perfect"`. If the user slouches or leans forward, the timer pauses and the system alerts them.
- **Progress Tracking:** Session statistics (e.g., "70% perfect posture, 20% neck lean, 10% back slouch") will be logged into a local CSV/JSON file to display a dashboard of user improvement over time.

## 5. Step-by-Step Implementation Guide
1. **Flash ESP32:** Write the Arduino/C++ code for the ESP32 to read MPU6050 tilt angles and broadcast "good" or "slouch" via Serial print.
2. **Serial Threading in Python:** Add a thread in the Python script to continuously read the ESP32's serial output without slowing down the MediaPipe model.
3. **Overlay UI Update:** Update the `cv2.putText()` functions to display the back status alongside the neck status.
4. **Implement Timer:** Add Python variables to track session duration and calculate final gamification scores.
5. **Two-Way Serial (Optional):** Update Python to write to the Serial port (`serial.write()`) to trigger the ESP32's vibration motor for neck leaning.
