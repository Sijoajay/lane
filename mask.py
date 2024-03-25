import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model

# Class to average lanes with
class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    """ Takes in a road image, re-sizes for the model, predicts the lane to be drawn from the model in G color, recreates an RGB image of a lane and merges with the original road image. """
    
    # Get image ready for feeding into model
    small_img = cv2.resize(image, (160, 80), interpolation=cv2.INTER_AREA)
    # small_img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    small_img = np.array(small_img, dtype=np.uint8)  # Convert to np.uint8
    small_img = small_img[None, :, :, :]  # Make prediction with neural network
    prediction = model.predict(small_img)[0] * 255  # Un-normalize value by multiplying by 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last 10 for average
    if len(lanes.recent_fit) > 10:
        lanes.recent_fit = lanes.recent_fit[1:]
    # Calculate weighted average detection
    weights = np.arange(10, 0, -1) / np.sum(np.arange(10, 0, -1))
    lanes.avg_fit = np.sum([w * p for w, p in zip(weights, lanes.recent_fit)], axis=0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    lane_image = lane_image.astype(np.uint8)  # Convert to np.uint8

    # Apply smoothing
    lane_image = cv2.GaussianBlur(lane_image, (9, 9), 0)  # Increase the kernel size for Gaussian blur
    lane_image = cv2.medianBlur(lane_image, 9)  # Apply median blur
    kernel = np.ones((7, 7), np.uint8)  # Increase the kernel size
    lane_image = cv2.morphologyEx(lane_image, cv2.MORPH_CLOSE, kernel)  # Try closing instead of opening

    # Check the number of channels in lane_image
    if lane_image.shape[-1] == 1:
        lane_image = cv2.cvtColor(lane_image, cv2.COLOR_GRAY2BGR)
    else:
        lane_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2BGR)

 
    image = image.astype(np.uint8)

   
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return result

if __name__ == '__main__':
  
    model = load_model('D:/DOWNLOADS/64convresnet40epchos.h5')

 
    lanes = Lanes()

    input_video = "C:/Users/sijoa/Desktop/lane/straight_lane.mp4"

  
    cap = cv2.VideoCapture(input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    while True:

        ret, frame = cap.read()
        if not ret:
            break


        processed_frame = road_lines(frame)


        cv2.imshow('Output', processed_frame)

        out.write(processed_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()


    cv2.destroyAllWindows()