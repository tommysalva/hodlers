# Import packages
import mediapipe as mp # Create a new environment if you get errors
import numpy as np
import cv2


# Function to create CNN input from input image
def create_input(image, max_num_hands = 1, min_detection_confidence = 0.1) :


    # INPUT
    # image as a numpy array ***we can change this depending on what streamlit outputs***
    
    # OUPUT
    # 28-by-28 image of hand as a numpy array

  # Load the image and convert to RGB
 # image = cv2.imread(image_path)
 # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  height, width, _ = image.shape # Save image dimensions
  # Create a zero-padded version of the image
  image_padded = np.zeros((height * 3, width * 3, 3))
  image_padded[height:2*height, width:2*width, :] = image

  # Initialize MediaPipe Hands object
  mp_hands = mp.solutions.hands.Hands(static_image_mode = True, # Can be changed to False if we do videos
                                      max_num_hands = max_num_hands,
                                      min_detection_confidence = min_detection_confidence)

  # Process the image to detect hands
  results = mp_hands.process(image)

  # Check if hands are detected
  if results.multi_hand_landmarks :
    x, y = [], [] # Store list of (x, y) coordinates
    for landmark in results.multi_hand_landmarks[0].landmark :
      x.append(int(landmark.x * width))
      y.append(int(landmark.y * height))
  else :
    print('No hands detected ...')
    return image, None, None, None, None, None

  # Release resources
  mp_hands.close()

  # Create box around the points with 2.5% extension
  x_min, x_max = int(min(x) - 2.5/100 * width),  int(max(x) + 2.5/100 * width)
  y_min, y_max = int(min(y) - 2.5/100 * height), int(max(y) + 2.5/100 * height)

  # Get box dimensions
  dx, dy = x_max - x_min, y_max - y_min
  diff = dy - dx

  # Extend dimension of smaller side of box to form a square
  if diff > 0 : # Height > Width
    if diff % 2 == 0 :
      x_min -= np.abs(diff) // 2
      x_max += np.abs(diff) // 2
    else :
      x_min -= (np.abs(diff) - 1) // 2
      x_max += (np.abs(diff) + 1) // 2
  elif diff < 0 : # Width > Height
    if diff % 2 == 0 :
      y_min -= np.abs(diff) // 2
      y_max += np.abs(diff) // 2
    else :
      y_min -= (np.abs(diff) - 1) // 2
      y_max += (np.abs(diff) + 1) // 2

  # Select subset of image
  hand = image_padded[height+y_min:height+y_max, width+x_min:width+x_max, :]
  # Downsample it to input size (28 x 28)
  hand_downsampled = np.array(cv2.resize(hand, (28, 28), interpolation = cv2.INTER_AREA).tolist())

  return hand_downsampled