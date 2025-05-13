import time
import threading
import cv2
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

# Thread control
running = True

# OpenCV optimization control variables
optimization_enabled = True
toggle_optimization = False
optimization_lock = threading.Lock()
optimization_toggle_interval = 30  # Toggle optimization every 30 seconds for testing
last_optimization_toggle = time.time()
auto_toggle_enabled = False  # Set to True to enable auto-toggling for testing

# Performance measurement
performance_data = {
    "frame_capture": {"fps": 0, "optimized": []},
    "black_detection": {"fps": 0, "optimized": []},
    "blue_detection": {"fps": 0, "optimized": []},
    "yellow_detection": {"fps": 0, "optimized": []},
    "movement_control": {"latency": 0, "optimized": []},
    "display": {"fps": 0, "optimized": []}
}
performance_lock = threading.Lock()

# Shared data structure for line detection results
line_results = {
    "black": {"detected": False, "angle": 0, "shift": 0, "area": 0, "mask": None, "timestamp": 0},
    "blue": {"detected": False, "angle": 0, "shift": 0, "area": 0, "mask": None, "timestamp": 0},
    "yellow": {"detected": False, "angle": 0, "shift": 0, "area": 0, "mask": None, "timestamp": 0}
}

# Shared frame for all detection threads to use
shared_frame = None
frame_ready = False

# Locks for thread synchronization
results_lock = threading.Lock()
frame_lock = threading.Lock()
frame_ready_lock = threading.Condition(frame_lock)

# Color priorities
COLOR_PRIORITY = {"black": 1, "blue": 2, "yellow": 3}

# Current active color for movement and display
active_color = "black"

# Pre-calculated HSV color ranges
COLOUR_RANGE = {
    "black": ((0, 0, 0), (179, 255, 98)),
    "blue": ((0, 142, 63), (38, 255, 138)),
    "yellow": ((58, 56, 143), (108, 249, 244))
}

# Predefine morphology kernels - create once for efficiency
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def move(left_speed, right_speed):
    """Control motor movement with specified speeds"""
    # Set motor direction pins based on speed sign
    GPIO.output(IN1, left_speed > 0)
    GPIO.output(IN2, left_speed < 0)
    GPIO.output(IN3, right_speed > 0)
    GPIO.output(IN4, right_speed < 0)
    
    # Set PWM duty cycles
    pwm_ENA.ChangeDutyCycle(abs(left_speed))
    pwm_ENB.ChangeDutyCycle(abs(right_speed))

def detect_line(frame, color):
    """Detect line using HSV color space and OpenCV functions"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Create mask with pre-defined color range
    lower_colour, upper_colour = COLOUR_RANGE[color]
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    
    # Apply morphological operations for noise reduction
    mask = cv2.morphologyEx(
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL),
        cv2.MORPH_OPEN, MORPH_KERNEL
    )
    
    # Find contours - use CHAIN_APPROX_SIMPLE for memory efficiency
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour using OpenCV's max function with contour area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return None
    
    # Calculate centroid
    cx = int(M["m10"] / M["m00"])
    
    # Calculate shift from center (avoid NumPy operations)
    frame_center = frame.shape[1] // 2
    shift = cx - frame_center
    
    # Calculate angle using OpenCV's fitLine 
    [vx, vy, _, _] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = cv2.fastAtan2(float(vy), float(vx))
    if angle > 180:
        angle -= 180
    
    # Calculate contour area directly with OpenCV
    area = cv2.contourArea(largest_contour)
    
    return {
        "detected": True,
        "angle": angle,
        "shift": shift,
        "mask": mask,
        "area": area
    }

def optimized_movement(angle, shift):
    """Determine movement based on line position with improved turning"""
    # Initialize turn state based on angle
    turn_state = 0
    if angle < 90 - TURN_ANGLE:
        turn_state = -1
    elif angle > 90 + TURN_ANGLE:
        turn_state = 1
    
    # Initialize shift state with more gradual response
    shift_state = 0
    if abs(shift) > SHIFT_MAX:
        shift_state = 1 if shift > 0 else -1
    elif abs(shift) > SHIFT_MAX/2:  # Add medium shift response
        shift_state = 0.5 if shift > 0 else -0.5
    
    # Calculate turn direction and value with more precision
    turn_dir = 0
    turn_val = 0
    
    # Prioritize shift correction for smoother movement
    if shift_state != 0:
        turn_dir = shift_state
        # Use proportional correction based on shift magnitude
        shift_correction = min(abs(shift) / 10, 3) * SHIFT_STEP
        turn_val = shift_correction
    elif turn_state != 0:
        turn_dir = turn_state
        turn_val = TURN_STEP
    
    # Determine motor speeds based on turn direction with more balanced values
    base_speed = 60  # Establish a consistent base speed
    
    if turn_dir < 0.5:  # Strong right turn
        return base_speed - 50, base_speed + 10, "Turn Right", turn_val
    elif turn_dir < 0:  # Gentle right turn
        return base_speed - 30, base_speed + 5, "Slight Right", turn_val
    elif turn_dir > -0.5:  # Strong left turn
        return base_speed + 10, base_speed - 50, "Turn Left", turn_val
    elif turn_dir > 0:  # Gentle left turn
        return base_speed + 5, base_speed - 30, "Slight Left", turn_val
    else:
        return base_speed, base_speed, "Straight", turn_val

def add_info_overlay(frame, color, turn_direction="", turn_value=0, optimization_status=True, fps=0):
    """Add information overlay to the frame using OpenCV operations"""
    # Set text color based on line color
    text_color = (255, 255, 255)  # Default white
    if color == "blue":
        text_color = (255, 200, 0)
    elif color == "yellow":
        text_color = (0, 0, 255)
    
    # Create overlay with rectangle - expanded for additional info
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Add text with single function calls
    cv2.putText(frame, f"Following: {color.upper()} line", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    if turn_direction:
        cv2.putText(frame, f"Movement: {turn_direction} ({turn_value})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Add optimization status
    opt_status = "ON" if optimization_status else "OFF"
    cv2.putText(frame, f"OpenCV Optimization: {opt_status}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Add FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return frame

def toggle_opencv_optimization():
    """Toggle OpenCV optimization and return current status"""
    global optimization_enabled
    with optimization_lock:
        optimization_enabled = not optimization_enabled
        cv2.setUseOptimized(optimization_enabled)
        current_status = cv2.useOptimized()
        print(f"OpenCV optimization set to: {current_status}")
    return current_status

def capture_frames():
    """Thread function to capture frames for all detection threads to use"""
    global running, shared_frame, frame_ready, frame_ready_lock, performance_data
    global last_optimization_toggle  # <- Add this here
    
    print("Starting frame capture thread")
    fps_counter = 0
    start_time = time.time()
    
    while running:
        try:
            # Check if optimization should be toggled
            if auto_toggle_enabled:
                current_time = time.time()
                if current_time - last_optimization_toggle >= optimization_toggle_interval:
                    last_optimization_toggle = current_time
                    is_optimized = toggle_opencv_optimization()
                    print(f"Auto-toggled OpenCV optimization to {is_optimized}")          
            # Capture new frame from camera
            new_frame = picam2.capture_array()
            
            # Update the shared frame with lock
            with frame_lock:
                shared_frame = new_frame
                frame_ready = True
                frame_ready_lock.notify_all()  # Notify all waiting threads
            
            # FPS calculation
            fps_counter += 1
            elapsed = time.time() - start_time
            if elapsed >= 5:
                fps = fps_counter / elapsed
                with performance_lock:
                    performance_data["frame_capture"]["fps"] = fps
                    performance_data["frame_capture"]["optimized"].append(
                        (fps, cv2.useOptimized())
                    )
                    if len(performance_data["frame_capture"]["optimized"]) > 10:
                        performance_data["frame_capture"]["optimized"].pop(0)
                
                print(f"Frame capture rate: {fps:.2f} FPS - Optimized: {cv2.useOptimized()}")
                fps_counter = 0
                start_time = time.time()
            
            # Short sleep to control capture rate
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
            # Wait for a new frame to be available
            with frame_lock:
                while not frame_ready and running:
                    frame_ready_lock.wait(0.1)
                
                if not running:
                    break
                    
                # Make a copy of the shared frame for processing
                current_frame = shared_frame.copy()
            
            # Detect line for this color
            result = detect_line(current_frame, color)
            
            # Update shared results with thread lock
            with results_lock:
                if result is not None:
                    line_results[color] = result
                    line_results[color]["timestamp"] = time.time()
                else:
                    line_results[color]["detected"] = False
                    line_results[color]["timestamp"] = time.time()
            
            # Performance tracking
            processed_counter += 1
            elapsed = time.time() - start_time
            if elapsed >= 5:
                fps = processed_counter / elapsed
                with performance_lock:
                    performance_data[f"{color}_detection"]["fps"] = fps
                    performance_data[f"{color}_detection"]["optimized"].append(
                        (fps, cv2.useOptimized())
                    )
                    if len(performance_data[f"{color}_detection"]["optimized"]) > 10:
                        performance_data[f"{color}_detection"]["optimized"].pop(0)
                
                print(f"{color} detection: {fps:.2f} FPS - Optimized: {cv2.useOptimized()}")
                processed_counter = 0
                start_time = time.time()
                
        except Exception as e:
            print(f"Error in {color} detection thread: {str(e)}")
            time.sleep(0.1)

def process_movement_control():
    """Thread function for robot movement based on detected lines"""
    global running, line_results, results_lock, active_color, performance_data
    
    print("Starting movement control thread")
    control_interval = 0.05  # 20Hz control rate
    last_update = time.time()
    control_counter = 0
    start_time = time.time()
    
    while running:
        try:
            current_time = time.time()
            
            # Update movement at fixed intervals
            if current_time - last_update >= control_interval:
                control_counter += 1
                start_control = time.time()
                last_update = current_time
                
                # Get latest results with thread lock
                with results_lock:
                    # Copy only necessary values instead of deep copying everything
                    detected_lines = {}
                    for color in line_results:
                        if line_results[color]["detected"]:
                            detected_lines[color] = {
                                "priority": COLOR_PRIORITY[color],
                                "angle": line_results[color]["angle"],
                                "shift": line_results[color]["shift"]
                            }
                
                # Apply movement based on highest priority detected line
                if detected_lines:
                    # Find highest priority color (most efficient method)
                    selected_color = max(detected_lines, key=lambda c: COLOR_PRIORITY[c])
                    selected_data = detected_lines[selected_color]
                    
                    # Update active color
                    active_color = selected_color
                    
                    # Calculate and apply movement
                    left_speed, right_speed, _, _ = optimized_movement(
                        selected_data["angle"], selected_data["shift"]
                    )
                    move(left_speed, right_speed)
                else:
                    # No line detected - reversing
                    move(-35, -30)
                
                # Measure control loop execution time
                control_time = time.time() - start_control
                
                # Performance tracking
                elapsed = time.time() - start_time
                if elapsed >= 5:
                    with performance_lock:
                        performance_data["movement_control"]["latency"] = control_time * 1000  # ms
                        performance_data["movement_control"]["optimized"].append(
                            (control_time * 1000, cv2.useOptimized())
                        )
                        if len(performance_data["movement_control"]["optimized"]) > 10:
                            performance_data["movement_control"]["optimized"].pop(0)
                    
                    print(f"Movement control: {control_counter / elapsed:.2f} Hz, "
                          f"Latency: {control_time*1000:.2f} ms - Optimized: {cv2.useOptimized()}")
                    control_counter = 0
                    start_time = time.time()
            
            # Short sleep to prevent CPU hogging
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in movement control thread: {str(e)}")
            time.sleep(0.1)

def process_display():
    """Thread function for visualization"""
    global running, line_results, results_lock, shared_frame, frame_lock, active_color, performance_data
    
    print("Starting display thread")
    display_interval = 0.033  # ~30 FPS
    last_display = time.time()
    display_counter = 0
    start_time = time.time()
    
    while running:
        try:
            current_time = time.time()
            
            # Update display at fixed intervals
            if current_time - last_display >= display_interval:
                display_counter += 1
                last_display = current_time
                
                # Get current frame with lock
                with frame_lock:
                    if shared_frame is None:
                        continue
                    display_frame = shared_frame.copy()
                
                # Get current results with lock
                with results_lock:
                    current_color = active_color
                    if line_results[current_color]["detected"]:
                        current_angle = line_results[current_color]["angle"]
                        current_shift = line_results[current_color]["shift"]
                        current_mask = line_results[current_color]["mask"]
                        is_detected = True
                    else:
                        is_detected = False
                
                # Process display elements
                if is_detected:
                    # Calculate movement info for display
                    _, _, movement_info, turn_val = optimized_movement(current_angle, current_shift)
                    
                    # Get current FPS and optimization status
                    with performance_lock:
                        fps = performance_data[f"{current_color}_detection"]["fps"]
                    
                    # Create overlay display
                    display_frame = add_info_overlay(
                        display_frame, 
                        current_color, 
                        movement_info, 
                        turn_val,
                        cv2.useOptimized(),
                        fps
                    )
                    
                    # Show mask if available
                    if current_mask is not None:
                        cv2.imshow("Line Mask", current_mask)
                else:
                    # No detection
                    display_frame = add_info_overlay(
                        display_frame, 
                        current_color, 
                        "No Line", 
                        0,
                        cv2.useOptimized(),
                        0
                    )
                
                # Show main display
                cv2.imshow("Line Follower", display_frame)
                
                # Performance tracking
                elapsed = time.time() - start_time
                if elapsed >= 5:
                    fps = display_counter / elapsed
                    with performance_lock:
                        performance_data["display"]["fps"] = fps
                        performance_data["display"]["optimized"].append(
                            (fps, cv2.useOptimized())
                        )
                        if len(performance_data["display"]["optimized"]) > 10:
                            performance_data["display"]["optimized"].pop(0)
                    
                    print(f"Display: {fps:.2f} FPS - Optimized: {cv2.useOptimized()}")
                    display_counter = 0
                    start_time = time.time()
                
                # Process key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('o'):
                    # Manual toggle of OpenCV optimization
                    is_optimized = toggle_opencv_optimization()
                    print(f"Manual toggle - OpenCV optimization: {is_optimized}")
                elif key == ord('a'):
                    # Toggle automatic optimization switching
                    global auto_toggle_enabled
                    auto_toggle_enabled = not auto_toggle_enabled
                    print(f"Auto-toggle optimization: {'enabled' if auto_toggle_enabled else 'disabled'}")
                elif key == ord('p'):
                    # Print performance comparison
                    print_performance_comparison()
            
            # Short sleep to prevent CPU hogging
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in display thread: {str(e)}")
            time.sleep(0.1)

def print_performance_comparison():
    """Print a comparison of performance with and without optimization"""
    with performance_lock:
        print("\n===== OpenCV Optimization Performance Comparison =====")
        for component, data in performance_data.items():
            print(f"\n{component.replace('_', ' ').title()}:")
            
            # Separate optimized and non-optimized data
            optimized_data = [item[0] for item in data["optimized"] if item[1]]
            non_optimized_data = [item[0] for item in data["optimized"] if not item[1]]
            
            if optimized_data and non_optimized_data:
                # Calculate averages
                opt_avg = sum(optimized_data) / len(optimized_data)
                non_opt_avg = sum(non_optimized_data) / len(non_optimized_data)
                
                # Print comparison
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
        # Initial OpenCV optimization check
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Initial OpenCV optimization status: {cv2.useOptimized()}")
        
        # Make sure optimization is enabled at startup
        cv2.setUseOptimized(True)
        optimization_enabled = True
        print(f"OpenCV optimization set to: {cv2.useOptimized()}")
        
        # Create threads
        frame_thread = threading.Thread(target=capture_frames)
        black_thread = threading.Thread(target=process_color_line, args=("black",))
        blue_thread = threading.Thread(target=process_color_line, args=("blue",))
        yellow_thread = threading.Thread(target=process_color_line, args=("yellow",))
        movement_thread = threading.Thread(target=process_movement_control)
        display_thread = threading.Thread(target=process_display)
        
        # Set as daemon threads
        all_threads = [frame_thread, black_thread, blue_thread, yellow_thread, 
                      movement_thread, display_thread]
        for thread in all_threads:
            thread.daemon = True
        
        # Start all threads
        for thread in all_threads:
            thread.start()
            # Small delay to stagger thread startup
            time.sleep(0.1)
        
        print("All threads started successfully")
        print("Press 'o' to manually toggle OpenCV optimization")
        print("Press 'a' to toggle automatic optimization switching")
        print("Press 'p' to print performance comparison")
        print("Press 'q' to quit")
        
        # Keep main thread running until interrupted
        while running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Program stopped by user")
        
    finally:
        # Signal threads to stop
        running = False
        
        # Clean up hardware
        pwm_ENA.stop()
        pwm_ENB.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        
        # Print final performance comparison
        print_performance_comparison()
        
        print("Cleanup complete")