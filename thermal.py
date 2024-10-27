import cv2
import numpy as np

def apply_thermal_colormap(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a color map to simulate thermal vision
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Capture the background frame
    ret, background = cap.read()
    if not ret:
        print("Failed to capture background frame.")
        return
    
    # Convert the background to grayscale and apply Gaussian blur
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the current frame to grayscale and apply Gaussian blur
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        # Compute the absolute difference between the background and current frame
        diff = cv2.absdiff(background_gray, gray_frame)
        
        # Apply thresholding to get the foreground mask
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply thermal colormap to the current frame
        thermal_frame = apply_thermal_colormap(frame)
        
        # Apply the mask to the thermal frame
        thermal_frame[thresh == 0] = 0
        
        # Display the resulting frame
        cv2.imshow('Thermal Background Subtraction', thermal_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
