import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use the correct index for your thermal camera

# Initialize the background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply the background subtractor
    fgMask = backSub.apply(frame)
    
    # Use a colormap to visualize the thermal effect
    thermal_effect = cv2.applyColorMap(fgMask, cv2.COLORMAP_JET)
    
    # Display the original frame and the thermal effect
    cv2.imshow('Original', frame)
    cv2.imshow('Thermal Effect', thermal_effect)
    
    # Exit if the user presses the 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the capture object and close the windows
cap.release()
cv2.destroyAllWindows()
