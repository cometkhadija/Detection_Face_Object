import cv2

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

### Step 1: Detect face from an image
# Read the input image
img = cv2.imread(r'C:/Users/ASUS/Desktop/Face_detect/test01.jpg')

# Check if image loaded
if img is None:
    print("Image not found or path is incorrect")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show image detection result
cv2.imshow('Face Detection (Image)', img)
print("Press any key to continue to webcam face detection...")
cv2.waitKey(0)
cv2.destroyAllWindows()

### Step 2: Detect face from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display result
    cv2.imshow('Face Detection (Webcam)', frame)

    # Press Esc to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
