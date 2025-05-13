import cv2 as cv
import numpy as np
import time
from picamera2 import Picamera2

# Constants
WINDOW_NAME = "HSV Color Calibration"

def nothing(x):
    """Empty callback function for trackbars"""
    pass

def create_calibration_window():
    """Create window with trackbars for HSV adjustment"""
    cv.namedWindow(WINDOW_NAME)
    
    # Create trackbars with better default values
    # H is typically 0-179 in OpenCV (not 0-255)
    cv.createTrackbar('H_min', WINDOW_NAME, 0, 179, nothing)
    cv.createTrackbar('S_min', WINDOW_NAME, 0, 255, nothing)
    cv.createTrackbar('V_min', WINDOW_NAME, 0, 255, nothing)
    cv.createTrackbar('H_max', WINDOW_NAME, 179, 179, nothing)
    cv.createTrackbar('S_max', WINDOW_NAME, 255, 255, nothing)
    cv.createTrackbar('V_max', WINDOW_NAME, 255, 255, nothing)

def main():
    # Create the calibration window with trackbars
    create_calibration_window()
    
    # Initialize PiCamera2
    try:
        picam2 = Picamera2()
        
        # Configure camera
        camera_config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(camera_config)
        
        # Start camera
        picam2.start()
        
        # Give camera time to initialize and adjust
        print("Initializing camera...")
        time.sleep(2)
        print("Camera ready!")
        
    except Exception as e:
        print(f"Error initializing PiCamera: {e}")
        return
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert from RGB (PiCamera) to BGR (OpenCV standard)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            
            # Get trackbar positions
            h_min = cv.getTrackbarPos('H_min', WINDOW_NAME)
            s_min = cv.getTrackbarPos('S_min', WINDOW_NAME)
            v_min = cv.getTrackbarPos('V_min', WINDOW_NAME)
            h_max = cv.getTrackbarPos('H_max', WINDOW_NAME)
            s_max = cv.getTrackbarPos('S_max', WINDOW_NAME)
            v_max = cv.getTrackbarPos('V_max', WINDOW_NAME)
            
            try:
                # Convert frame to HSV
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                
                # Define color range
                lower_boundary = np.array([h_min, s_min, v_min])
                upper_boundary = np.array([h_max, s_max, v_max])
                
                # Create mask and apply it
                mask = cv.inRange(hsv, lower_boundary, upper_boundary)
                result = cv.bitwise_and(frame, frame, mask=mask)
                
                # Add text showing current HSV values
                text = f"Lower HSV: [{h_min}, {s_min}, {v_min}]"
                cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
                
                text = f"Upper HSV: [{h_max}, {s_max}, {v_max}]"
                cv.putText(frame, text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
                
                # Show additional information
                text = "Press 'q' to quit, 's' to save values"
                cv.putText(frame, text, (10, frame.shape[0] - 20), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display results
                cv.imshow("Original", frame)
                cv.imshow("Mask", mask)
                cv.imshow(WINDOW_NAME, result)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
            
            # Check for key presses
            key = cv.waitKey(1) & 0xFF
            
            # Exit on 'q' key
            if key == ord('q'):
                break
                
            # Save values on 's' key
            elif key == ord('s'):
                filename = f"hsv_values_{time.strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w') as f:
                    f.write(f"# HSV Values saved on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"lower_color = np.array([{h_min}, {s_min}, {v_min}])\n")
                    f.write(f"upper_color = np.array([{h_max}, {s_max}, {v_max}])\n")
                print(f"HSV values saved to {filename}")
    
    except KeyboardInterrupt:
        print("Program interrupted by user")
    
    finally:
        # Cleanup
        print("Closing camera and windows...")
        picam2.stop()
        cv.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()