import threading
import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize variables
counter = 0
face_match = False
lock = threading.Lock()  # For thread-safe operations on face_match
reference_img = cv2.imread("C://Users//91799//Downloads//reference.jpg")

# Function to check face matching
def check_face(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, reference_img.copy())
        with lock:
            face_match = result['verified']
    except Exception as e:
        with lock:
            face_match = False

# Main loop
while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:  # Process every 30 frames
            thread = threading.Thread(target=check_face, args=(frame.copy(),))
            thread.start()
        counter += 1

        # Display result on the frame
        with lock:
            if face_match:
                cv2.putText(frame, 'MATCH!', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'NOT MATCH!', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
