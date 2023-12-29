import cv2

# Create a variable to assign the Haar Cascade Classifier file (full body)
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Capture video
cap = cv2.VideoCapture('walking.avi')

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Convert each frame into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass each frame to the classifier
        bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

        # Draw rectangle around detected bodies
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with rectangles around detected bodies
        cv2.imshow('Pedestrian Detection', frame)

        # Check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
