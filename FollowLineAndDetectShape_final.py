import time
import threading
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Define motor pins
IN1, IN2, IN3, IN4 = 20, 21, 19, 26
ENA, ENB = 12, 13

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Setup GPIO pins
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# PWM setup for motor speed control
pwm_ENA = GPIO.PWM(ENA, 100)
pwm_ENB = GPIO.PWM(ENB, 100)
pwm_ENA.start(0)
pwm_ENB.start(0)

# Configuration parameters
TURN_ANGLE = 60  # degrees
SHIFT_MAX = 40  # pixels
SHIFT_STEP = 10  # base shift correction
TURN_STEP = 12   # base turn correction
PAUSE_DURATION = 3  # seconds to pause when shape detected
RESUME_DURATION = 3  # seconds to resume line following, ignoring shapes

# Thread control
running = True
resuming = False  # Track resume period after pause
resume_start_time = 0  # Timestamp for resume period start

# OpenCV optimization control variables
optimization_enabled = True
optimization_lock = threading.Lock()
optimization_toggle_interval = 30  # Toggle optimization every 30 seconds for testing
last_optimization_toggle = time.time()
auto_toggle_enabled = False  # Set to True to enable auto-toggling for testing

# Performance measurement
performance_data = {
    "frame_capture": {"fps": 0, "optimized": []},
    "black_detection": {"fps": 0, "optimized": []},
    "blue_detection": {"fps": 0, "optimized": []},
    "red_detection": {"fps": 0, "optimized": []},
    "shape_detection": {"fps": 0, "optimized": []},
    "movement_control": {"latency": 0, "optimized": []},
    "display": {"fps": 0, "optimized": []}
}
performance_lock = threading.Lock()

# Shared data structures
line_results = {
    "black": {"detected": False, "angle": 0, "shift": 0, "area": 0, "mask": None, "timestamp": 0},
    "blue": {"detected": False, "angle": 0, "shift": 0, "area": 0, "mask": None, "timestamp": 0},
    "red": {"detected": False, "angle": 0, "shift": 0, "area": 0, "mask": None, "timestamp": 0}
}
shape_results = {
    "detected": False,
    "type": None,  # 'shape'
    "name": None,  # e.g., 'Circle'
    "color": None,  # e.g., 'red', 'blue', 'green'
    "location": None,  # (x, y) top-left corner
    "shape": None,  # (height, width) for bounding box
    "timestamp": 0
}

# Shared frame for all detection threads
shared_frame = None
frame_ready = False

# Locks for thread synchronization
results_lock = threading.Lock()
shape_lock = threading.Lock()
frame_lock = threading.Lock()
frame_ready_lock = threading.Condition(frame_lock)

# Color priorities
COLOR_PRIORITY = {"black": 1, "blue": 2, "red": 3}

# Current active color for movement and display
active_color = "black"

# Pre-calculated HSV color ranges for lines
COLOUR_RANGE = {
    "black": ((0, 0, 0), (179, 255, 98)),
    "blue": ((0, 142, 63), (38, 255, 138)),
    "red": ((113, 32, 80), (172, 255, 230))
}

# Color ranges for shape detection (red, blue, green)
SHAPE_COLOUR_RANGE = {
    "blue": ((100, 100, 100), (140, 255, 255)),  # Hue around 120° (blue)
    "green": ((40, 100, 100), (80, 255, 255)),   # Hue around 60° (green)
    "red1": ((0, 100, 100), (10, 255, 255)),     # Hue near 0° (red lower)
    "red2": ((170, 100, 100), (180, 255, 255))   # Hue near 180° (red upper)
}

# Predefine morphology kernels
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def move(left_speed, right_speed):
    """Control motor movement with specified speeds"""
    GPIO.output(IN1, left_speed > 0)
    GPIO.output(IN2, left_speed < 0)
    GPIO.output(IN3, right_speed > 0)
    GPIO.output(IN4, right_speed < 0)
    pwm_ENA.ChangeDutyCycle(abs(left_speed))
    pwm_ENB.ChangeDutyCycle(abs(right_speed))

def detect_line(frame, color):
    """Detect line using HSV color space and OpenCV functions"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_colour, upper_colour = COLOUR_RANGE[color]
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL),
        cv2.MORPH_OPEN, MORPH_KERNEL
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    frame_center = frame.shape[1] // 2
    shift = cx - frame_center
    [vx, vy, _, _] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = cv2.fastAtan2(float(vy), float(vx))
    if angle > 180:
        angle -= 180
    area = cv2.contourArea(largest_contour)
    return {
        "detected": True,
        "angle": angle,
        "shift": shift,
        "mask": mask,
        "area": area
    }

def detect_shape(contour):
    """Detects shapes based on contour properties with improved triangle detection"""
    area = cv2.contourArea(contour)
    if area < 500:
        return "Unknown"
    
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "Unknown"
    
    epsilon = 0.03 * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Unknown"
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    if vertices == 3 and solidity > 0.8:
        if 0.5 <= aspect_ratio <= 2.0:
            return "Triangle"
    
    if circularity > 0.85:
        return "Circle"
    
    if 0.4 <= circularity <= 0.85:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if len(hull_indices) > 2 and len(contour) > len(hull_indices):
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        if depth > 15:
                            return "Pacman"
            except cv2.error:
                pass
    
    if solidity < 0.95:
        return "Unknown"
    if vertices == 4:
        return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    
    return "Unknown"

def optimized_movement(angle, shift):
    """Determine movement based on line position"""
    turn_state = 0
    if angle < 90 - TURN_ANGLE:
        turn_state = -1
    elif angle > 90 + TURN_ANGLE:
        turn_state = 1
    shift_state = 0
    if abs(shift) > SHIFT_MAX:
        shift_state = 1 if shift > 0 else -1
    elif abs(shift) > SHIFT_MAX/2:
        shift_state = 0.5 if shift > 0 else -0.5
    turn_dir = 0
    turn_val = 0
    if shift_state != 0:
        turn_dir = shift_state
        shift_correction = min(abs(shift) / 10, 3) * SHIFT_STEP
        turn_val = shift_correction
    elif turn_state != 0:
        turn_dir = turn_state
        turn_val = TURN_STEP
        
    base_speed = 50
    
    if turn_dir < 0.5:
        return base_speed - 40, base_speed + 10, "Turn Right", turn_val
    elif turn_dir < 0:
        return base_speed - 30, base_speed + 5, "Slight Right", turn_val
    elif turn_dir > -0.5:
        return base_speed + 10, base_speed - 40, "Turn Left", turn_val
    elif turn_dir > 0:
        return base_speed + 5, base_speed - 30, "Slight Left", turn_val
    else:
        return base_speed, base_speed, "Straight", turn_val

def add_info_overlay(frame, color, turn_direction="", turn_value=0, optimization_status=True, fps=0, shape_info=None, paused=False):
    """Add information overlay to the frame"""
    text_color = (255, 255, 255)
    if color == "blue":
        text_color = (255, 200, 0)
    elif color == "red":
        text_color = (0, 0, 255)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Following: {color.upper()} line", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    movement_text = "Paused" if paused else turn_direction
    cv2.putText(frame, f"Movement: {movement_text} ({turn_value})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    opt_status = "ON" if optimization_status else "OFF"
    cv2.putText(frame, f"OpenCV Optimization: {opt_status}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    if shape_info:
        cv2.putText(frame, f"Detected: {shape_info}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def toggle_opencv_optimization():
    """Toggle OpenCV optimization"""
    global optimization_enabled
    with optimization_lock:
        optimization_enabled = not optimization_enabled
        cv2.setUseOptimized(optimization_enabled)
        current_status = cv2.useOptimized()
        print(f"OpenCV optimization set to: {current_status}")
    return current_status

def capture_frames():
    """Thread function to capture frames"""
    global running, shared_frame, frame_ready, frame_ready_lock, performance_data, last_optimization_toggle
    print("Starting frame capture thread")
    fps_counter = 0
    start_time = time.time()
    while running:
        try:
            if auto_toggle_enabled:
                current_time = time.time()
                if current_time - last_optimization_toggle >= optimization_toggle_interval:
                    last_optimization_toggle = current_time
                    is_optimized = toggle_opencv_optimization()
                    print(f"Auto-toggled OpenCV optimization to {is_optimized}")
            new_frame = picam2.capture_array()
            with frame_lock:
                shared_frame = new_frame
                frame_ready = True
                frame_ready_lock.notify_all()
            fps_counter += 1
            elapsed = time.time() - start_time
            if elapsed >= 5:
                fps = fps_counter / elapsed
                with performance_lock:
                    performance_data["frame_capture"]["fps"] = fps
                    performance_data["frame_capture"]["optimized"].append((fps, cv2.useOptimized()))
                    if len(performance_data["frame_capture"]["optimized"]) > 10:
                        performance_data["frame_capture"]["optimized"].pop(0)
                print(f"Frame capture rate: {fps:.2f} FPS - Optimized: {cv2.useOptimized()}")
                fps_counter = 0
                start_time = time.time()
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in frame capture thread: {str(e)}")
            time.sleep(0.1)

def process_color_line(color):
    """Thread function to process a specific color line detection"""
    global running, shared_frame, frame_ready, frame_ready_lock, line_results, results_lock, performance_data
    print(f"Starting {color} line detection thread")
    processed_counter = 0
    start_time = time.time()
    while running:
        try:
            with frame_lock:
                while not frame_ready and running:
                    frame_ready_lock.wait(0.1)
                if not running:
                    break
                current_frame = shared_frame.copy()
            result = detect_line(current_frame, color)
            with results_lock:
                if result is not None:
                    line_results[color] = result
                    line_results[color]["timestamp"] = time.time()
                else:
                    line_results[color]["detected"] = False
                    line_results[color]["timestamp"] = time.time()
            processed_counter += 1
            elapsed = time.time() - start_time
            if elapsed >= 5:
                fps = processed_counter / elapsed
                with performance_lock:
                    performance_data[f"{color}_detection"]["fps"] = fps
                    performance_data[f"{color}_detection"]["optimized"].append((fps, cv2.useOptimized()))
                    if len(performance_data[f"{color}_detection"]["optimized"]) > 10:
                        performance_data[f"{color}_detection"]["optimized"].pop(0)
                print(f"{color} detection: {fps:.2f} FPS - Optimized: {cv2.useOptimized()}")
                processed_counter = 0
                start_time = time.time()
        except Exception as e:
            print(f"Error in {color} detection thread: {str(e)}")
            time.sleep(0.1)

def process_shape():
    """Thread function to detect colored shapes (red, blue, green)"""
    global running, shared_frame, frame_ready, frame_ready_lock, shape_results, shape_lock, performance_data
    print("Starting shape detection thread")
    processed_counter = 0
    start_time = time.time()
    while running:
        try:
            with frame_lock:
                while not frame_ready and running:
                    frame_ready_lock.wait(0.05)
                if not running:
                    break
                current_frame = shared_frame.copy()
            # Downscale frame for detection
            scale_factor = 0.5
            small_frame = cv2.resize(current_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HSV)

            # Create color masks
            mask_blue = cv2.inRange(hsv, SHAPE_COLOUR_RANGE["blue"][0], SHAPE_COLOUR_RANGE["blue"][1])
            mask_green = cv2.inRange(hsv, SHAPE_COLOUR_RANGE["green"][0], SHAPE_COLOUR_RANGE["green"][1])
            mask_red1 = cv2.inRange(hsv, SHAPE_COLOUR_RANGE["red1"][0], SHAPE_COLOUR_RANGE["red1"][1])
            mask_red2 = cv2.inRange(hsv, SHAPE_COLOUR_RANGE["red2"][0], SHAPE_COLOUR_RANGE["red2"][1])
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # Apply morphological operations
            for mask in [mask_blue, mask_green, mask_red]:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)

            # Find contours in each mask
            largest_contours = []
            for color, mask in [("blue", mask_blue), ("green", mask_green), ("red", mask_red)]:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    largest_contours.append((largest, color))

            symbol_detected = False
            if largest_contours:
                # Select the largest contour across all colors
                overall_largest, detected_color = max(largest_contours, key=lambda item: cv2.contourArea(item[0]))
                area = cv2.contourArea(overall_largest)
                if area >= 500:
                    shape_name = detect_shape(overall_largest)
                    if shape_name != "Unknown":
                        x, y, w, h = cv2.boundingRect(overall_largest)
                        # Scale coordinates back to original resolution
                        x_orig, y_orig = int(x / scale_factor), int(y / scale_factor)
                        w_orig, h_orig = int(w / scale_factor), int(h / scale_factor)
                        symbol_detected = True
                        with shape_lock:
                            shape_results.update({
                                "detected": True,
                                "type": "shape",
                                "name": shape_name,
                                "color": detected_color,
                                "location": (x_orig, y_orig),
                                "shape": (h_orig, w_orig),
                                "timestamp": time.time()
                            })
                        print(f"Detected {detected_color} shape: {shape_name}")

            if not symbol_detected:
                with shape_lock:
                    shape_results.update({
                        "detected": False,
                        "type": None,
                        "name": None,
                        "color": None,
                        "location": None,
                        "shape": None,
                        "timestamp": time.time()
                    })

            processed_counter += 1
            elapsed = time.time() - start_time
            if elapsed >= 5:
                fps = processed_counter / elapsed
                with performance_lock:
                    performance_data["shape_detection"]["fps"] = fps
                    performance_data["shape_detection"]["optimized"].append((fps, cv2.useOptimized()))
                    if len(performance_data["shape_detection"]["optimized"]) > 10:
                        performance_data["shape_detection"]["optimized"].pop(0)
                print(f"Shape detection: {fps:.2f} FPS - Optimized: {cv2.useOptimized()}")
                processed_counter = 0
                start_time = time.time()
        except Exception as e:
            print(f"Error in shape detection thread: {str(e)}")
            time.sleep(0.05)

def process_movement_control():
    """Thread function for robot movement"""
    global running, line_results, results_lock, shape_results, shape_lock, active_color, performance_data, resuming, resume_start_time
    print("Starting movement control thread")
    control_interval = 0.05
    last_update = time.time()
    control_counter = 0
    start_time = time.time()
    pausing = False
    pause_start_time = 0
    while running:
        try:
            current_time = time.time()
            if current_time - last_update >= control_interval:
                control_counter += 1
                start_control = time.time()
                last_update = current_time
                with shape_lock:
                    shape_detected = shape_results["detected"]
                    shape_timestamp = shape_results["timestamp"]
                
                if resuming:
                    if current_time - resume_start_time >= RESUME_DURATION:
                        resuming = False
                        print("Resume period ended, shape detection re-enabled")
                elif shape_detected and not pausing:
                    if current_time - shape_timestamp < 0.5:
                        pausing = True
                        pause_start_time = current_time
                        move(0, 0)
                        print(f"Shape detected, pausing for {PAUSE_DURATION} seconds")
                
                if pausing:
                    if current_time - pause_start_time >= PAUSE_DURATION:
                        pausing = False
                        resuming = True
                        resume_start_time = current_time
                        print(f"Resuming line following for {RESUME_DURATION} seconds, ignoring shapes")
                    else:
                        move(0, 0)
                        continue
                
                with results_lock:
                    detected_lines = {}
                    for color in line_results:
                        if line_results[color]["detected"]:
                            detected_lines[color] = {
                                "priority": COLOR_PRIORITY[color],
                                "angle": line_results[color]["angle"],
                                "shift": line_results[color]["shift"]
                            }
                if detected_lines:
                    selected_color = min(detected_lines, key=lambda c: COLOR_PRIORITY[c])
                    selected_data = detected_lines[selected_color]
                    active_color = selected_color
                    left_speed, right_speed, _, _ = optimized_movement(selected_data["angle"], selected_data["shift"])
                    move(left_speed, right_speed)
                else:
                    move(-35, -30)
                
                control_time = time.time() - start_control
                elapsed = time.time() - start_time
                if elapsed >= 5:
                    with performance_lock:
                        performance_data["movement_control"]["latency"] = control_time * 1000
                        performance_data["movement_control"]["optimized"].append((control_time * 1000, cv2.useOptimized()))
                        if len(performance_data["movement_control"]["optimized"]) > 10:
                            performance_data["movement_control"]["optimized"].pop(0)
                    print(f"Movement control: {control_counter / elapsed:.2f} Hz, "
                          f"Latency: {control_time*1000:.2f} ms - Optimized: {cv2.useOptimized()}")
                    control_counter = 0
                    start_time = time.time()
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in movement control thread: {str(e)}")
            time.sleep(0.1)

def process_display():
    """Thread function for visualization"""
    global running, line_results, results_lock, shape_results, shape_lock, shared_frame, frame_lock, active_color, performance_data
    print("Starting display thread")
    display_interval = 0.033
    last_display = time.time()
    display_counter = 0
    start_time = time.time()
    while running:
        try:
            current_time = time.time()
            if current_time - last_display >= display_interval:
                display_counter += 1
                last_display = current_time
                with frame_lock:
                    if shared_frame is None:
                        continue
                    display_frame = shared_frame.copy()
                with results_lock:
                    current_color = active_color
                    if line_results[current_color]["detected"]:
                        current_angle = line_results[current_color]["angle"]
                        current_shift = line_results[current_color]["shift"]
                        current_mask = line_results[current_color]["mask"]
                        is_line_detected = True
                    else:
                        is_line_detected = False
                with shape_lock:
                    shape_detected = shape_results["detected"]
                    shape_type = shape_results["type"]
                    shape_name = shape_results["name"]
                    shape_color = shape_results["color"]
                    shape_location = shape_results["location"]
                    shape_shape = shape_results["shape"]
                    shape_timestamp = shape_results["timestamp"]
                paused = shape_detected and (current_time - shape_timestamp < PAUSE_DURATION)
                shape_info = None
                if shape_detected and shape_location and shape_shape:
                    h, w = shape_shape
                    x, y = shape_location
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)
                    label = f"{shape_color.capitalize()} {shape_name}"
                    shape_info = f"Shape: {label}"
                    cv2.putText(display_frame, label, (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if is_line_detected:
                    _, _, movement_info, turn_val = optimized_movement(current_angle, current_shift)
                    with performance_lock:
                        fps = performance_data[f"{current_color}_detection"]["fps"]
                    display_frame = add_info_overlay(
                        display_frame, current_color, movement_info, turn_val,
                        cv2.useOptimized(), fps, shape_info, paused
                    )
                    if current_mask is not None:
                        cv2.imshow("Line Mask", current_mask)
                else:
                    display_frame = add_info_overlay(
                        display_frame, current_color, "No Line", 0,
                        cv2.useOptimized(), 0, shape_info, paused
                    )
                cv2.imshow("Line Follower", display_frame)
                elapsed = time.time() - start_time
                if elapsed >= 5:
                    fps = display_counter / elapsed
                    with performance_lock:
                        performance_data["display"]["fps"] = fps
                        performance_data["display"]["optimized"].append((fps, cv2.useOptimized()))
                        if len(performance_data["display"]["optimized"]) > 10:
                            performance_data["display"]["optimized"].pop(0)
                    print(f"Display: {fps:.2f} FPS - Optimized: {cv2.useOptimized()}")
                    display_counter = 0
                    start_time = time.time()
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('o'):
                    is_optimized = toggle_opencv_optimization()
                    print(f"Manual toggle - OpenCV optimization: {is_optimized}")
                elif key == ord('a'):
                    global auto_toggle_enabled
                    auto_toggle_enabled = not auto_toggle_enabled
                    print(f"Auto-toggle optimization: {'enabled' if auto_toggle_enabled else 'disabled'}")
                elif key == ord('p'):
                    print_performance_comparison()
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in display thread: {str(e)}")
            time.sleep(0.1)

def print_performance_comparison():
    """Print performance comparison"""
    with performance_lock:
        print("\n===== OpenCV Optimization Performance Comparison =====")
        for component, data in performance_data.items():
            print(f"\n{component.replace('_', ' ').title()}:")
            optimized_data = [item[0] for item in data["optimized"] if item[1]]
            non_optimized_data = [item[0] for item in data["optimized"] if not item[1]]
            if optimized_data and non_optimized_data:
                opt_avg = sum(optimized_data) / len(optimized_data)
                non_opt_avg = sum(non_optimized_data) / len(non_optimized_data)
                if "latency" in component:
                    print(f"  Optimized latency: {opt_avg:.2f} ms")
                    print(f"  Non-optimized latency: {non_opt_avg:.2f} ms")
                    print(f"  Improvement: {(non_opt_avg - opt_avg) / non_opt_avg * 100:.1f}%")
                else:
                    print(f"  Optimized FPS: {opt_avg:.2f}")
                    print(f"  Non-optimized FPS: {non_opt_avg:.2f}")
                    print(f"  Improvement: {(opt_avg - non_opt_avg) / non_opt_avg * 100:.1f}%")
            else:
                print("  Not enough data for comparison")
        print("\n================================================")

if __name__ == "__main__":
    try:
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Initial OpenCV optimization status: {cv2.useOptimized()}")
        cv2.setUseOptimized(True)
        optimization_enabled = True
        print(f"OpenCV optimization set to: {cv2.useOptimized()}")
        frame_thread = threading.Thread(target=capture_frames)
        black_thread = threading.Thread(target=process_color_line, args=("black",))
        blue_thread = threading.Thread(target=process_color_line, args=("blue",))
        red_thread = threading.Thread(target=process_color_line, args=("red",))
        shape_thread = threading.Thread(target=process_shape)
        movement_thread = threading.Thread(target=process_movement_control)
        display_thread = threading.Thread(target=process_display)
        all_threads = [frame_thread, black_thread, blue_thread, red_thread, 
                       shape_thread, movement_thread, display_thread]
        for thread in all_threads:
            thread.daemon = True
        for thread in all_threads:
            thread.start()
            time.sleep(0.1)
        print("All threads started successfully")
        print("Press 'o' to manually toggle OpenCV optimization")
        print("Press 'a' to toggle automatic optimization switching")
        print("Press 'p' to print performance comparison")
        print("Press 'q' to quit")
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        running = False
        pwm_ENA.stop()
        pwm_ENB.stop()
        GPIO.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()
        print_performance_comparison()
        print("Cleanup complete")