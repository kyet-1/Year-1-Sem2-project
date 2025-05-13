import cv2
import numpy as np
from picamera2 import Picamera2

# Define paths to template images (adjust paths for Raspberry Pi filesystem)
template_paths = { 
    'right_arrow': '/home/pi/symbols/right_arrow.jpeg',
    'down_arrow': '/home/pi/symbols/down_arrow.jpeg',
    'left_arrow': '/home/pi/symbols/left_arrow.jpeg',
    'up_arrow': '/home/pi/symbols/up_arrow.jpeg',
    'measure_distance': '/home/pi/symbols/measure_distance.jpeg',
    'face_recognition': '/home/pi/symbols/face_recognition.jpeg',
    'traffic_stop': '/home/pi/symbols/traffic_stop.jpeg',
    'no_entry': '/home/pi/symbols/no_entry.jpeg'
}

# Load templates in grayscale and categorize them
templates = {}
non_arrow_templates = {}  # Priority group
arrow_templates = {}      # Secondary group

for name, path in template_paths.items():
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template '{name}' from {path}")
    templates[name] = template
    if 'arrow' in name:
        arrow_templates[name] = template
    else:
        non_arrow_templates[name] = template

def detect_shape(contour):
    """Detects one of six shapes based on contour properties."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "Unknown"
    
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Unknown"
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Pacman: circle with a quarter missing
    if 0.5 <= circularity <= 0.85:
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 2 and len(contour) > len(hull):
            try:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        if depth > 20:
                            return "Pacman"
            except cv2.error:
                pass
    
    # Ensure shape is solid (except Pacman)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < 0.95:
        return "Unknown"
    
    # Circle: high circularity
    if circularity > 0.85:
        return "Circle"
    
    # Polygon shapes based on vertices
    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    
    return "Unknown"

def multi_scale_template_match(image, template, scales=[1.0, 1.25, 1.5]):
    """Perform template matching at multiple scales and return the best match with scale."""
    best_score = -1.0
    best_location = None
    best_template_shape = None
    best_scale = None
    
    for scale in scales:
        scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scaled_image.shape[0] < template.shape[0] or scaled_image.shape[1] < template.shape[1]:
            continue
        
        result = cv2.matchTemplate(scaled_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_location = (int(max_loc[0] / scale), int(max_loc[1] / scale))
            best_template_shape = (int(template.shape[0] / scale), int(template.shape[1] / scale))
            best_scale = scale
    
    return best_score, best_location, best_template_shape, best_scale

# Initialize Picamera2
picam2 = Picamera2()
# Configure the camera for preview (adjust resolution as needed)
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Threshold for a valid match
threshold = 0.7

# Dictionary to log detected scales for analysis
scale_log = {name: [] for name in templates.keys()}

try:
    while True:
        # Capture frame from Picamera2
        frame = picam2.capture_array()
        if frame is None:
            print("Error: Could not capture frame.")
            break
        
        # Convert to grayscale for processing
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # --- Multi-Scale Template Matching with Priority ---
        best_score = 0.0
        best_location = None
        best_template_name = None
        best_template_shape = None
        best_scale = None
        symbol_detected = False

        # Step 1: Check non-arrow symbols first (highest priority)
        for name, template in non_arrow_templates.items():
            if template is None:
                continue
            
            score, location, shape, scale = multi_scale_template_match(frame_gray, template)
            if score >= threshold and score > best_score:
                best_score = score
                best_location = location
                best_template_name = name
                best_template_shape = shape
                best_scale = scale
                symbol_detected = True

        # Step 2: Check arrow symbols only if no non-arrow symbols are found
        if not symbol_detected:
            for name, template in arrow_templates.items():
                if template is None:
                    continue
                
                score, location, shape, scale = multi_scale_template_match(frame_gray, template)
                if score >= threshold and score > best_score:
                    best_score = score
                    best_location = location
                    best_template_name = name
                    best_template_shape = shape
                    best_scale = scale
                    symbol_detected = True

        # Step 3: Draw and log if a symbol is detected
        if symbol_detected:
            h, w = best_template_shape
            top_left = best_location
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            label = f"{best_template_name} (Scale: {best_scale})"
            cv2.putText(frame, label, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Log the scale for this template
            scale_log[best_template_name].append(best_scale)
            print(f"Detected {best_template_name} at scale {best_scale} with score {best_score:.2f}")
        
        # Step 4: Shape detection only if no symbols are detected
        else:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                
                shape = detect_shape(contour)
                if shape == "Unknown":
                    continue
                
                # Draw contour in green
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(frame, shape, (cX - 30, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Scale Identification with Priority', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the camera and release resources
    picam2.stop()
    cv2.destroyAllWindows()
    
    # Analyze logged scales
    print("\nScale Analysis:")
    for name, scales in scale_log.items():
        if scales:  # Only analyze if there were detections
            avg_scale = np.mean(scales)
            std_scale = np.std(scales)
            print(f"{name}: Detected {len(scales)} times, Avg Scale = {avg_scale:.2f}, Std Dev = {std_scale:.2f}")
        else:
            print(f"{name}: No detections")
