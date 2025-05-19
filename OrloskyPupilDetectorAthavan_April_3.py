import cv2
import numpy as np
import tkinter as tk
import os
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import subprocess
import json
#import RPi.GPIO as GPIO
import time
from scipy.optimize import curve_fit
import threading
#from picamera2 import Picamera2, Preview
#from picamera2.encoders import H264Encoder

fixation_points = []  # Global list to store fixation points (during calibration test)
pupil_positions = []  # Global list to store pupil positions
pupil_diameters = []  # Global list to store pupil diameters
timestamps = [] # Global variable to store timestamps

#Screen coordinates for 9-point calibration
lcd_targets = {
    "Top Left": (320, 180), "Top Center": (960, 180), "Top Right": (1600, 180),
    "Middle Left": (320, 540), "Middle Center": (960, 540), "Middle Right": (1600, 540),
    "Bottom Left": (320, 900), "Bottom Center": (960, 900), "Bottom Right": (1600, 900),
}

# Screen points for 3x3 grid (adding extra point due to window closing at last point)
screen_points = np.array([
    [102, 60], [511.5, 60], [921, 60],
    [102, 300], [511.5, 300], [921, 300],
    [102, 540], [511.5, 540], [921, 540], [921, 540]
])

def display_grid_points():
    # Create a blank white image
    screen_width = 1024
    screen_height = 600
    blank_frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

    # Create a fullscreen window
    window_name = "Grid Points"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Iterate through each point in the grid
    for i, point in enumerate(screen_points):
        # Create a new blank frame for each point
        frame = blank_frame.copy()

        # Draw the current point as a black circle (others remain invisible)
        cv2.circle(frame, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)

        # Record timestamp for this point
        timestamps.append((time.time(), f"Point {i+1}: {point}"))

        # Display the frame
        cv2.imshow(window_name, frame)

        # Add a small delay to ensure proper rendering of the first frame
        if i == 0:
            cv2.waitKey(100)  # Wait for a short duration to refresh the window

        # Wait for 2 seconds before moving to the next point
        time.sleep(2)

        # If this is the last point, explicitly wait and then close the window
        if i == len(screen_points) - 1:
            cv2.destroyAllWindows()

        # Break if 'q' is pressed during display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        print(point)

    # Close the OpenCV window
    cv2.destroyAllWindows()
        
#Helps map fixation points from user to screen coordinates
def calibrate_screen(video_path, input_method, fixation_points):
    eye_points = []
    fixation_data = []
    
    # Initialize video capture
    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
    elif input_method == 2:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Camera input
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    for index, (key, value) in enumerate(lcd_targets.items()):
        point = lcd_targets[key]
        print(f"Please fixate on {lcd_targets[key]}")
        
        # Display fixation point on the screen
        blank_frame = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.circle(blank_frame, point, 10, (0, 255, 0), -1)  # Green circle for fixation point
        cv2.imshow("Calibration", blank_frame)

        pupil_positions_for_point = []  # Store pupil positions for this fixation point

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, pupil_center = process_frame(frame, fixation_data)
            if pupil_center:
                pupil_positions_for_point.append(pupil_center)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # Press 'n' to move to the next fixation point
                break
            elif key == ord('q'):  # Press 'q' to quit calibration
                cap.release()
                cv2.destroyAllWindows()
                return

        # Store mapping of fixation point to average pupil position
        if pupil_positions_for_point:
            avg_pupil_position = np.mean(pupil_positions_for_point, axis=0)
            #fixation_data.append((point, tuple(avg_pupil_position)))
            fixation_data.append(tuple(avg_pupil_position))

    cap.release()
    cv2.destroyAllWindows()

    print("Calibration complete!")
    print("Fixation Data:", fixation_data)
    return fixation_data
    
def pupil_diameter_plot_to_time(df):
    # Plot diameter over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['frame']/60, df['diameter'])
    plt.title('Pupil Diameter Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Diameter')
    plt.grid(True)
    plt.show()
    
def compute_pupil_diameter_metrics(df, fps, output_path, eye_side):
    # Convert frames to time
    df['time'] = df['frame'] / fps

    #plot_df = df

    #df = df[df['time'] >= df['time'].max()-2]

    min_pupil_diam = df['diameter'].min()
    min_diameter_time = df[df['diameter'] == min_pupil_diam]['time'].iloc[0]

    max_pupil_diam = df[df['time'] < min_diameter_time]['diameter'].max()
    percent_constriction = ((max_pupil_diam - min_pupil_diam) / max_pupil_diam) * 100
    
    #Computes when minimum diameter is found
    max_diameter_time = df[df['diameter'] == max_pupil_diam]['time'].iloc[0]
    
    min_diameter_frame = df[df['diameter'] == min_pupil_diam]['frame'].iloc[0]
    max_diameter_frame = df[df['diameter'] == max_pupil_diam]['frame'].iloc[0]

    filtered_df = df[df['time'] >= max_diameter_time]

    # Compute derivative (velocity)
    df['velocity'] = filtered_df['diameter'].diff() / filtered_df['time'].diff()

    # Find peak constriction and dilation velocity
    peak_constriction_velocity = df['velocity'].min()
    peak_constriction_velocity_time = df[df['velocity'] == peak_constriction_velocity]['time'].iloc[0]

    peak_dilation_velocity = df['velocity'].max()
    peak_dilation_velocity_time = df[df['velocity'] == peak_dilation_velocity]['time'].iloc[0]

    # Compute average constriction velocity (from max to min)
    constriction_phase = df[(df['diameter'] >= min_pupil_diam) & (df['time'] <= min_diameter_time) & ((df['time'] >= max_diameter_time))]
    average_constriction_velocity = (max_pupil_diam - min_pupil_diam) / (constriction_phase['time'].iloc[-1] - constriction_phase['time'].iloc[0])

    # Compute average dilation velocity (from min to recovery)
    recovery_threshold = min_pupil_diam + 0.25 * (max_pupil_diam - min_pupil_diam) #5.6
    print(recovery_threshold)
    #recovery_time = df[df['diameter'] >= recovery_threshold]['time'].iloc[0]
    
    if df[(df['diameter'] >= recovery_threshold) & (df['time'] <= min_diameter_time) & (df['time'] >= max_diameter_time)].empty:
        recovery_time = None  # or np.nan
        time_to_redilation = None
    else:
        #recovery_time = df[(df['diameter'] >= recovery_threshold) & (df['time'] <= min_diameter_time) & (df['time'] >= max_diameter_time)]['time'].iloc[0]
        # Find the index of the closest diameter value to recovery_threshold
        time_df = df[df['time'] <= min_diameter_time]
        closest_index = (time_df['diameter'] - recovery_threshold).abs().idxmin()

        # Get the corresponding time value
        recovery_time = df.loc[closest_index, 'time']
        print(recovery_time)
        time_to_redilation = recovery_time - max_diameter_time
    
    dilation_phase = df[(df['diameter'] >= min_pupil_diam) & (df['time'] >= min_diameter_time)]
    #avg_dilation_velocity = (recovery_threshold - min_diameter) / (recovery_time - df[df['diameter'] == min_diameter]['time'].iloc[0])
    average_dilation_velocity = (recovery_threshold - min_pupil_diam) / (dilation_phase['time'].iloc[-1] - dilation_phase['time'].iloc[0])
    
    
    # Print results
    print(f"Max Pupillary Diameter: {max_pupil_diam}")
    print(f"Min Pupillary Diameter: {min_pupil_diam}")
    
    print(f"Time when Max Pupillary Diameter Occurs: {max_diameter_time}")
    print(f"Time when Min Pupillary Diameter Occurs: {min_diameter_time}")
    print(f"Frame when Max Pupillary Diameter Occurs: {max_diameter_frame}")
    print(f"Frame when Min Pupillary Diameter Occurs: {min_diameter_frame}")
    
    
    print(f"Percentage Constriction: {percent_constriction:.2f}%")
    print(f"Peak Constriction Velocity: {peak_constriction_velocity:.4f} mm/s")

    print(f"Time when Peak Constriction Velocity Occurs: {peak_constriction_velocity_time}")

    print(f"Average Constriction Velocity: {average_constriction_velocity:.4f} mm/s")
    print(f"Peak Dilation Velocity: {peak_dilation_velocity:.4f} mm/s")

    print(f"Time when Peak Dilation Velocity Occurs: {peak_dilation_velocity_time}")
    print(f"Average Dilation Velocity: {average_dilation_velocity:.4f} mm/s")
    if time_to_redilation is not None:
        print(f"Time to 75% Pupillary Recovery: {time_to_redilation:.4f} s")
    
    #Indicates specific test metrics are from
    direction_eye = 'l'
    if eye_side == 'right':    
        direction_eye = 'r'
    
    # Store results in a dictionary
    metrics = {
        f"max_pupil_diam_{direction_eye}": max_pupil_diam,
        f"min_pupil_diam_{direction_eye}": min_pupil_diam,
        f"percent_contstriction_{direction_eye}": round(percent_constriction, 2),
        f"peak_constriction_velocity_{direction_eye}": round(peak_constriction_velocity, 4),
        f"average_constriction_velocity_{direction_eye}": round(average_constriction_velocity, 4),
        f"peak_dilation_velocity_{direction_eye}": round(peak_dilation_velocity, 4),
        f"average_dilation_velocity_{direction_eye}": round(average_dilation_velocity, 4),
        f"time_to_redilation_{direction_eye}": round(time_to_redilation, 4) if time_to_redilation is not None else "Not Found",
    }
    
    plr_file_path = os.path.join('PLR', f"plr_metrics_{eye_side}.json")

    # Save to a file
    with open(plr_file_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4)

    #Keep dataframe to only that of 6 seconds or less
    #filtered_df = df[df['time'] < 6]

    #random_df = df[df['time'] > 25]

    # Plot the pupil diameter over time
    plt.figure(figsize=(10, 5)) 
    plt.plot(df['time'], df['diameter'], label='Pupil Diameter', color='b')
    if recovery_threshold is not None:
        plt.axhline(y=recovery_threshold, color='g', linestyle='--', label='75% Recovery Threshold')
    if recovery_time is not None:
        plt.axvline(x=recovery_time, color='r', linestyle='--', label='75% Recovery Time')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pupil Diameter (mm)")
    plt.title("Pupil Diameter Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    
    return metrics

#ORIGINAL FUNCTION
def sinusoidal_compare_pupil_to_sine(df, output_path, eye_side, test_num):
    # Extract pupil data
    time = df['frame']
    pupil_y = df['y']
    
    # Normalize time to [0, 1]
    time_norm = (time - time.min()) / (time.max() - time.min())
    
    # Calculate smoothed_sine value 
    normalized = (pupil_y - pupil_y.min()) / (pupil_y.max() - pupil_y.min())
    
    smoothed_sine = normalized
    
    # Generate sine wave data
    sine_x_raw = np.linspace(0, 1, len(time))
    
    sine_x = (sine_x_raw - sine_x_raw.min()) / (sine_x_raw.max() - sine_x_raw.min())

    if test_num == 1:
        # Normalized to [0, 1]
        sine_y_raw = 150 + 150 * np.sin(4 * np.pi * sine_x)
    else:
        sine_y_raw = 150 + 150 * np.sin(8 * np.pi * sine_x)
   
    sine_y = (sine_y_raw - sine_y_raw.min()) / (sine_y_raw.max() - sine_y_raw.min())
    
    common_x = np.linspace(0, 1, 1000)
    
    # Interpolate pupil data to match sine wave x-values
    pupil_interp = interp1d(time_norm, smoothed_sine, kind='cubic', fill_value="extrapolate")
    pupil_y_interp = pupil_interp(common_x)
    
    ref_interp = interp1d(sine_x, sine_y, kind='cubic')
    ref_y_interp = ref_interp(common_x)

    # Calculate phase lag
    correlation = correlate(pupil_y_interp - np.mean(pupil_y_interp), 
                             ref_y_interp - np.mean(ref_y_interp), 
                            mode='full', method='auto')
    lags = np.arange(-len(common_x) + 1, len(common_x))
    lag = lags[np.argmax(correlation)]
    phase_lag = lag / len(common_x)  # Normalize lag to [0, 1] range
    
    # Convert phase lag to degrees
    phase_lag_degrees = phase_lag * 360
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(common_x, ref_y_interp, label='Sine Wave: y = 0.5 + 0.5*sin(2πx)', color='blue')
    plt.plot(common_x, pupil_y_interp, label='Smoothed Pupil Y Position', color='red')
    plt.title(f'Comparison of Smoothed Pupil Y Position to Sine Wave (Phase Lag: {phase_lag_degrees:.2f}°)')
    plt.xlabel('Normalized Time')
    plt.ylabel('Normalized Vertical Position')
    plt.legend()
    plt.grid(True)
    #plt.show()
    
    plt.savefig(output_path)
    plt.close()
    
    # Calculate similarity metrics
    mse = mean_squared_error(pupil_y_interp, ref_y_interp)
    correlation, _ = pearsonr(pupil_y_interp, ref_y_interp)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Pearson Correlation Coefficient: {correlation:.4f}")
    print(f"Phase Lag: {phase_lag_degrees:.2f}°")
    
    #Indicates specific test metrics are from
    direction_eye = 'l'
    degree_test = 180
    if eye_side == 'right':    
        direction_eye = 'r'
        
    if test_num == 2:
        degree_test = 360
    
    # Convert results to JSON format
    sinusoid_smp_results_json = {
        f"mean_squared_error_{direction_eye}_{degree_test}": mse,
        f"pearson_coefficient_{direction_eye}_{degree_test}": correlation,
        f"phase_lag_{direction_eye}_{degree_test}": phase_lag_degrees
    }
    
    smp_file_path = os.path.join('sine_smooth_pursuit', f"sinusoid_smp_results_{eye_side}_test{test_num}.json")

    # Save to a JSON file or return as a string
    with open(smp_file_path, "w") as json_file:
        json.dump(sinusoid_smp_results_json, json_file, indent=4)

    return sinusoid_smp_results_json


def horizontal_compare_pupil_to_reference_sine(df_pupil, output_path, eye_side):
    # Extract data from pupil tracking
    pupil_x_coords = df_pupil['frame']
    pupil_y_coords = df_pupil['x']  # Use x-coordinate directly
    
    # Generate reference sine wave
    ref_x_coords = np.linspace(0, 4, len(pupil_x_coords))
    
    ref_y_coords = 200 * np.sin((5/3) * (np.pi) * ref_x_coords)
    
    # Normalize data
    def normalize_data(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    pupil_x_normalized = normalize_data(pupil_x_coords)
    pupil_y_normalized = normalize_data(pupil_y_coords)
    ref_x_normalized = normalize_data(ref_x_coords)
    ref_y_normalized = normalize_data(ref_y_coords)
    
    smoothed_sine = pupil_y_normalized
    
    # Interpolate to common x-axis
    common_x = np.linspace(0, 1, 1000)
    pupil_interp = interp1d(pupil_x_normalized, smoothed_sine, kind='cubic')
    ref_interp = interp1d(ref_x_normalized, ref_y_normalized, kind='cubic')
    
    pupil_y_interp = pupil_interp(common_x)
    ref_y_interp = ref_interp(common_x)
 
    # Compare the waves
    horizontal_compare_waves_custom(common_x, pupil_y_interp, ref_y_interp, output_path, eye_side)

def horizontal_compare_waves_custom(x, pupil_wave, ref_wave, output_path, eye_side):
    plt.figure(figsize=(10, 6))
    plt.plot(x, pupil_wave, label='Pupil X-coordinate (Normalized)', color='blue')
    plt.plot(x, ref_wave, label='Reference Sine Wave (Normalized)', color='red', linestyle='dashed')
    plt.title('Comparison of Normalized Pupil X-coordinate and Reference Sine Wave')
    plt.xlabel('Normalized Time')
    plt.ylabel('Normalized Horizontal Position')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path)
    plt.close()
    #plt.show()
    
    # Compute similarity metrics
    mse = mean_squared_error(pupil_wave, ref_wave)
    #correlation = np.corrcoef(pupil_wave, ref_wave)[0, 1]
    
    # Calculate phase lag
    correlation = correlate(pupil_wave - np.mean(pupil_wave), 
                             ref_wave - np.mean(ref_wave), 
                            mode='full', method='auto')
    lags = np.arange(-len(x) + 1, len(x))
    lag = lags[np.argmax(correlation)]
    phase_lag = lag / len(x)  # Normalize lag to [0, 1] range
    
    # Convert phase lag to degrees
    phase_lag_degrees = phase_lag * 360
    
    pearson_corr, _ = pearsonr(pupil_wave, ref_wave)
    
    print(f"Mean Squared Error: {mse}")
    print(f"Pearson Correlation Coefficient: {pearson_corr}")
    print(f"Phase Lag: {phase_lag_degrees:.3f} degrees")
    
    # Convert results to JSON format
    horizontal_smp_results_json = {
        "mean_squared_error": mse,
        "pearson_correlation_coefficient": pearson_corr,
        "phase_lag": phase_lag_degrees
    }
    
    smp_file_path = os.path.join('horizontal_smooth_pursuit', f"horizontal_smp_results_{eye_side}.json")

    # Save to a JSON file or return as a string
    with open(smp_file_path, "w") as json_file:
        json.dump(horizontal_smp_results_json, json_file, indent=4)

# Remove outliers and interpolate missing values
def remove_outliers(series, n_std=2):
    mean = series.mean()
    std = series.std()
    distance_from_mean = abs(series - mean)
    return series.mask(distance_from_mean > n_std * std)

def plr_interpolate_and_smooth(pupil_diameters):
    if not pupil_diameters:
       return None

    frames = range(len(pupil_diameters))

    df = pd.DataFrame({'frame': frames, 'diameter': pupil_diameters})

    df['diameter'] = remove_outliers(df['diameter']).interpolate(method='akima').bfill().ffill()

    # Ensure lengths match after interpolation
    if len(df['diameter']) != len(df['frame']):
        raise ValueError("Length mismatch after interpolation")

    # Smooth using Savitzky-Golay filter while ensuring window_length is valid
    window_length = min(71, len(df['diameter']) - (len(df['diameter']) % 2 == 0))  # Must be odd and <= length of data
    if window_length < 3:  # Minimum window length for savgol_filter
        window_length = 3

    df['diameter'] = savgol_filter(df['diameter'], window_length=window_length, polyorder=2)
    df['diameter'] = gaussian_filter1d(df['diameter'], sigma=2)

    return df

def sm_pursuit_interpolate_and_smooth(pupil_positions):
    if not pupil_positions:
        return None

    frames = range(len(pupil_positions))
    x_coords, y_coords = zip(*pupil_positions)

    df = pd.DataFrame({'frame': frames, 'x': x_coords, 'y': y_coords})

    df['x'] = remove_outliers(df['x']).interpolate(method='akima').bfill().ffill()
    df['y'] = remove_outliers(df['y']).interpolate(method='akima').bfill().ffill()

    # Ensure lengths match after interpolation
    if len(df['x']) != len(df['frame']) or len(df['y']) != len(df['frame']):
        raise ValueError("Length mismatch after interpolation")

    # Smooth using Savitzky-Golay filter while ensuring window_length is valid
    window_length = min(21, len(df['x']) - (len(df['x']) % 2 == 0))  # Must be odd and <= length of data
    if window_length < 3:  # Minimum window length for savgol_filter
        window_length = 3

    df['x'] = savgol_filter(df['x'], window_length=window_length, polyorder=2)
    df['x'] = gaussian_filter1d(df['x'], sigma=2)
    
    df['y'] = savgol_filter(df['y'], window_length=window_length, polyorder=2)
    df['y'] = gaussian_filter1d(df['y'], sigma=2)

    return df

def plot_pupil_positions(df, output_path):
    if df is None or df.empty:
        print("No data available for plotting.")
        return

    plt.figure(figsize=(15, 6))

    # Plot X positions
    plt.subplot(1, 2, 1)
    plt.plot(df['frame'], df['x'], 'b-', linewidth=2)
    plt.title('Pupil X Positions Over Time (Smoothed)')
    plt.xlabel('Frame')
    plt.ylabel('X Position (pixels)')
    plt.grid(True)


    # Plot Y positions
    plt.subplot(1, 2, 2)
    plt.plot(df['frame'], df['y'], 'r-', linewidth=2)
    plt.title('Pupil Y Positions Over Time (Smoothed)')
    plt.xlabel('Frame')
    plt.ylabel('Y Position (pixels)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    #plt.show()

# Crop the image to maintain a specific aspect ratio (width:height) before resizing. 
def crop_to_aspect_ratio(image, width=640, height=480):
    
    # Calculate current aspect ratio
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset+new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset+new_height, :]

    return cv2.resize(cropped_img, (width, height))

#apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    # Calculate the threshold as the sum of the two input values
    threshold = darkestPixelValue + addedThreshold
    # Apply the binary threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_image

#Finds a square area of dark pixels in the image
#@param I input image (converted to grayscale during search process)
#@return a point within the pupil region
def get_darkest_area(image):

    ignoreBounds = 20 #don't search the boundaries of the image for ignoreBounds pixels
    imageSkipSize = 10 #only check the darkness of a block for every Nth x and y pixel (sparse sampling)
    searchArea = 20 #the size of the block to search
    internalSkipSize = 5 #skip every Nth x and y pixel in the local search area (sparse sampling)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    min_sum = float('inf')
    darkest_point = None

    # Loop over the image with spacing defined by imageSkipSize, ignoring the boundaries
    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            # Calculate sum of pixel values in the search area, skipping pixels based on internalSkipSize
            current_sum = np.int64(0)
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            # Update the darkest point if the current block is darker
            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)  # Center of the block

    return darkest_point

#mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    # Create a mask initialized to black
    mask = np.zeros_like(image)

    # Calculate the top-left corner of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)

    # Calculate the bottom-right corner of the square
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    # Set the square area in the mask to white
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
   
def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

#returns the largest contour that is not extremely long or tall
#contours is the list of contours, pixel_thresh is the max pixels to filter, and ratio_thresh is the max ratio
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)

            # Calculate the length-to-width ratio and width-to-length ratio
            length_to_width_ratio = length / width
            width_to_length_ratio = width / length

            # Pick the higher of the two ratios
            current_ratio = max(length_to_width_ratio, width_to_length_ratio)

            # Check if highest ratio is within the acceptable threshold
            if current_ratio <= ratio_thresh:
                # Update the largest contour if the current one is bigger
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    # Return a list with only the largest contour, or an empty list if no contour was found
    if largest_contour is not None:
        return [largest_contour]
    else:
        return []

#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

def process_frames(thresholded_image_strict, thresholded_image_medium, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window, position_list, diameter_list):
  
    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = [] #holds an array of the best contour points from the fitting process
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2]
    final_goodness = 0
    
    # Initialize pupil_diameter with a default value
    pupil_diameter = None
    
    #iterate through binary images and see which fits the ellipse best
    for i in range(1,3):
        # Dilate the binary image
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            
            ellipse = cv2.fitEllipse(reduced_contours[0])

            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])

            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]

            #show intermediary images with text output
            if debug_mode_on:
                cv2.putText(gray_copies[i-1], "%filled:     " + str(current_goodness[0])[:5] + " (percentage of filled contour pixels inside ellipse)", (10,30), font, .55, (255,255,255), 1) #%filled
                cv2.putText(gray_copies[i-1], "abs. pix:   " + str(total_pixels[0]) + " (total pixels under fit ellipse)", (10,50), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "pix ratio:  " + str(total_pixels[1]) + " (total pix under fit ellipse / contour border pix)", (10,70), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "final:     " + str(final_goodness) + " (filled*ratio)", (10,90), font, .55, (255,255,255), 1) #skewedness
                cv2.imshow(name_array[i-1] + " threshold", image_array[i-1])
                cv2.imshow(name_array[i-1], gray_copies[i-1])

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image

    if debug_mode_on:
        cv2.imshow("Reduced contours of best thresholded image", ellipse_reduced_contours)
    
    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0] > 5):
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse

        # Get the major and minor axis lengths
        (major_axis, minor_axis) = ellipse[1]
        
        # Set a threshold for the ellipse size (adjust as needed)
        size_threshold = 170  # in pixels
        
        # Check if the ellipse size exceeds the threshold
        if major_axis > size_threshold or minor_axis > size_threshold:
            # Set a fixed size for the pupil outline
            fixed_size = (150, 170)  # adjust as needed
            #fixed_size = (major_axis*0.5, minor_axis*0.8)
            # Calculate the center of the ellipse
            center = tuple(map(int, ellipse[0]))
            pupil_diameter = (fixed_size[0] + fixed_size[1]) / 2
            # Draw the fixed-size ellipse
            cv2.ellipse(test_frame, (center, fixed_size, ellipse[2]), (55, 255, 0), 2)
        else:
            pupil_diameter = (major_axis + minor_axis) / 2
            # Draw the original ellipse
            cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)

    if pupil_diameter is not None:
        #Convert pupil diameter from pixels to millimetres (according to DPI equation)
        #new_pupil_diameter = pupil_diameter/25.4
        new_pupil_diameter = pupil_diameter/23.785
    else:
        new_pupil_diameter = 0
    
    # Display pupil diameter on frame if it is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (10, 30)  # Position of text on the frame
    text_color = (255, 255, 255)  # White color for text
    text_scale = 0.7
    text_thickness = 2
    
    diameter_list.append(new_pupil_diameter)
            
    if new_pupil_diameter != 0:
        cv2.putText(test_frame,
                    f"Pupil Diameter: {new_pupil_diameter:.2f} mm",
                    text_position,
                    font,
                    text_scale,
                    text_color,
                    text_thickness)
    else:
        cv2.putText(test_frame,
                    "No Pupil Detected",
                    text_position,
                    font,
                    text_scale,
                    text_color,
                    text_thickness)
        
    if render_cv_window:
        cv2.imshow('best_thresholded_image_contours_on_frame', test_frame)
        
    # Create an empty image to draw contours
    contour_img3 = np.zeros_like(image_array[i-1])
    
    if len(final_contours[0]) >= 5:
        contour = np.array(final_contours[0], dtype=np.int32).reshape((-1, 1, 2)) #format for cv2.fitEllipse
        ellipse = cv2.fitEllipse(contour) # Fit ellipse
        cv2.ellipse(gray_frame, ellipse, (255,255,255), 2)  # Draw with white color and thickness of 2

    if final_contours and len(final_contours[0]) >= 5:
        ellipse = cv2.fitEllipse(final_contours[0])
        center_x, center_y = map(int, ellipse[0])
        #position_list.append((center_x, center_y))  # Store X and Y positions
        return final_rotated_rect, (center_x, center_y)
    else:
        return final_rotated_rect, None


# Finds the pupil in an individual frame and returns the center point
def process_frame(frame, position_list, diameter_list):
    
    debug_mode_on = False

    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)

    #find the darkest point
    darkest_point = get_darkest_area(frame)
    
    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 7)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250) 
    
    #take the three images thresholded at different levels and process them
    final_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, frame, gray_frame, darkest_point, False, False, position_list, diameter_list)
    
    return final_rotated_rect

def choose_comparison_method():
    print("Choose a comparison method:")
    print("1. Sinusoidal compare pupil to sine")
    print("2. Horizontal compare pupil to reference sine")
    
    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice == '1' or choice == '2':
            return int(choice)
        else:
            print("Invalid choice. Please enter 1 or 2.")

# Loads a video and finds the pupil in each frame
def process_video(video_path, input_method, comparison_method, position_list, diameter_list, dir):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter('output_video.mp4', fourcc, 120.0, (640, 480))  # Output video filename, codec, frame rate, and frame size

    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second is: {fps}")
    elif input_method == 2:
        cap = cv2.VideoCapture(00, cv2.CAP_DSHOW)  # Camera input
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print((width, height))
    #cap = cv2.VideoCapture(video_path)

    '''
    def count_frames_manually(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        cap.release()
        
        return frame_count

    total_frames_old = count_frames_manually(video_path)
    print(total_frames_old)

    # Get the total number of frames and the frame rate
    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30

    # Calculate the starting frame for the last 5 seconds
    start_frame = max(0, total_frames_old - int(fps * 5))

    # Set the position of the video capture to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    '''
    
    debug_mode_on = False
    
    temp_center = (0,0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the region of interest (ROI) (Modify these values as needed)
        
        #Right Eye
        if dir == 'right':
            x_start, y_start, width, height = 325, 25, 600, 350  # Example ROI coordinates
            #original: 200 50 600 350
            #changed x_start from 250 to 275
        else:
        #Left Eye
            x_start, y_start, width, height = 0, 260, 500, 500  # Example ROI coordinates
            #original: 200 250 600 300
            

        # Crop the frame to the selected ROI
        frame = frame[y_start:y_start+height, x_start:x_start+width]

        _, pupil_center = process_frame(frame, position_list, diameter_list)
        if pupil_center:
            position_list.append(pupil_center)  # Append valid positions
        
        # Crop and resize frame
        frame = crop_to_aspect_ratio(frame)

        #find the darkest point
        darkest_point = get_darkest_area(frame)

        # Convert to grayscale to handle pixel value operations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        
        # apply thresholding operations at different levels
        # at least one should give us a good ellipse segment
        thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
        thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

        thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 7)#medium
        thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
        
        #take the three images thresholded at different levels and process them
        pupil_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, frame, gray_frame, darkest_point, debug_mode_on, True, position_list, diameter_list)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d') and debug_mode_on == False:  # Press 'q' to start debug mode
            debug_mode_on = True
        elif key == ord('d') and debug_mode_on == True:
            debug_mode_on = False
            cv2.destroyAllWindows()
        if key == ord('q'):  # Press 'q' to quit
            out.release()
            break   
        elif key == ord(' '):  # Press spacebar to start/stop
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Press spacebar again to resume
                    break
                elif key == ord('q'):  # Press 'q' to quit
                    break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    df = sm_pursuit_interpolate_and_smooth(position_list)
    diameter_df = plr_interpolate_and_smooth(diameter_list)
    os.makedirs('PLR', exist_ok=True)
    os.makedirs('sine_smooth_pursuit', exist_ok=True)
    file_path = os.path.join('PLR', 'left_eye_plot.png')

    output_path = os.path.join('sine_smooth_pursuit', 'left_eye_plot_test_1.png')

    sine_metrics = sinusoidal_compare_pupil_to_sine(df, output_path, 'left', 1)

    #pupil_diameter_plot_to_time(diameter_df)
    #metrics = compute_pupil_diameter_metrics(diameter_df, 60, file_path, 'left')
    
    return df, diameter_df
    
def convert_timestamps_to_relative(input_file, output_file):
    # Read the original timestamps file
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Parse the first timestamp to use as the reference point
    first_timestamp = float(lines[0].split(" - ")[0])

    # Create a list to store the converted timestamps
    relative_timestamps = []

    for line in lines:
        # Split the line into timestamp and screen point information
        unix_timestamp, screen_point = line.split(" - ")
        unix_timestamp = float(unix_timestamp)

        # Convert to relative timestamp by subtracting the first timestamp
        relative_timestamp = unix_timestamp - first_timestamp

        # Format and store the converted line
        relative_timestamps.append(f"{relative_timestamp:.6f} - {screen_point}")

    # Write the converted timestamps to a new file
    with open(output_file, "w") as f:
        f.writelines(relative_timestamps)

def extract_segments(video_path, timings, output_dir, codec='mp4v', file_ext='.mp4'):
    """
    Split a video into segments based on start/end times from a CSV.
    
    Args:
        video_path (str): Path to input video (e.g., "left_eye.h264")
        timings_csv (str): Path to CSV with columns: [start_time, end_time, ...]
        output_dir (str): Directory to save segments (e.g., "left_eye_segments")
        codec (str): Video codec (e.g., 'mp4v', 'XVID')
        file_ext (str): Output file extension (e.g., '.mp4', '.avi')
    """
    # Create output directory if missing
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read timings (expects CSV with start_time/end_time columns)
    #timings = pd.read_csv(timings_csv)
    
    for idx, row in timings.iterrows():
        if idx == 0:
            start_time = float(row['start_time']) 
        else:
            start_time = float(row['start_time']) + 1.0
        end_time = float(row['end_time']) + 1.0

        print((start_time, end_time))
        segment_name = f"point_{idx}{file_ext}"
        output_path = os.path.join(output_dir, segment_name)
    # Calculate frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        print((start_frame, end_frame))
        # Set start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Extract frames
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            current_frame += 1
        
        out.release()
        print(f"Saved segment {idx}: {output_path} ({end_time-start_time:.2f}s)")
    
    cap.release()

#Prompts the user to select a video file if the hardcoded path is not found
def select_video():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    video_path = 'magic.mp4'
    if not os.path.exists(video_path):
        print("No file found at hardcoded path. Please select a video file.")
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.h264")])
        if not video_path:
            print("No file selected. Exiting.")
            return
        
    #ADD CALIBRATION FUNCTION CODE!
    
    comparison_method = choose_comparison_method()
    
    #Will run to perform test (currently just smooth pursuit test)
    #second parameter is 1 for video 2 for webcam
    df, diameter_df = process_video(video_path, 1, comparison_method, pupil_positions, pupil_diameters, "left")
    #fixation_points = df[['x', 'y']].to_numpy()
    #print(np.mean(fixation_points, axis=0))
    

if __name__ == "__main__":
    #display_grid_points() #Starts calibration

    #timings = pd.read_csv("calibration_timings_noemie_trial_6.csv")

    #extract_segments( "right_eye_calibration_noemie_trial_7.h264", timings, "split_videos")

    select_video()

    '''
    gaze_point_list = []

    for num in range(9):
        pos_data = []
        diam_data = []
        df, diameter_df = process_video(f"./split_videos/point_{num}.mp4", 1, 1, pos_data, diam_data, "right")
        fixation_points = df[['x', 'y']].to_numpy()
        mean_fixation_point = np.mean(fixation_points, axis=0)
        gaze_point_list.append(mean_fixation_point)

    gaze_points = np.array(gaze_point_list)
    print(gaze_points)
    print(screen_points[0:9])

    # Create a polynomial model
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

    # Apply RANSAC
    ransac = RANSACRegressor(poly_model, min_samples=0.5, residual_threshold=50, random_state=42)
    ransac.fit(gaze_points, screen_points[0:9])
    '''
    
    #can use ransac.predict() on pupil positions to better map to screen (for smooth pursuit tests)
