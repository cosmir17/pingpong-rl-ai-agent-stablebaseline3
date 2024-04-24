import numpy as np
import platform
from PIL import Image
from mss import mss
from game_stage import GameStage
import tensorflow as tf
from tensorflow import keras
from skimage.metrics import structural_similarity as ssim
import random
import cv2


os_name = platform.system()
# print(os_name)
monitor = {"top": 30, "left": 40, "width": 1133, "height": 702}  # using Magnet on Mac
if os_name == 'Linux' or os_name == 'Darwin':
    monitor = {"top": 55, "left": 1130, "width": 550, "height": 400}
elif os_name == "Windows":
    from window_grabber import *
    place_resize_nq_window()
    monitor = {"top": 0, "left": 0, "width": 1261, "height": 702}

game_stuck_without_ball = cv2.imread("game_stuck_without_ball.png")
game_start_page_loaded = cv2.imread("game_start_page_loaded.png")
    
    
def capture_window():
    with mss() as sct:
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        img = tf.image.rgb_to_grayscale(img)
        return img
        

def which_stage(img):
    top_left = (143, 17)
    bottom_right = (956, 773)
    game_stuck_without_ball_cropped = game_stuck_without_ball[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    current_centre_image = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    current_centre_image = current_centre_image.numpy()
    current_centre_image = current_centre_image.squeeze()
    current_centre_image = (current_centre_image * 255).astype(np.uint8)
    
    mse_value = mse(game_stuck_without_ball_cropped, current_centre_image)
    img_cut_in_half = img[:img.shape[0]//2, :, :]
    img_cut_in_half_np = img_cut_in_half.numpy()
    img_cut_in_half_np = np.squeeze(img_cut_in_half_np) # remove last dimension

    game_start_page_loaded_resized = cv2.resize(game_start_page_loaded, (img_cut_in_half.shape[1], img_cut_in_half.shape[0]))
    game_start_page_loaded_gray = cv2.cvtColor(game_start_page_loaded_resized, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(game_start_page_loaded_gray, img_cut_in_half_np)
    if ssim_score > 0.8:
        return GameStage(1) #hold screen
    else:
        if mse_value < 2995:
            return GameStage(2) #stuck_without_ball
        else:
            return GameStage(0) #in progress
# 2987 -> no ball
# 2994 -> ball stuck right bottom

# 2998 -> game before start page

def capture_display():
    with mss() as sct:
        num = str(random.randint(0, 9434734734))
        filename = "screen_is_black_" + num + ".png"
        sct.shot(output=filename)
        return filename


def convert_raw_scren_to_tf_np(sct_img):
    img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
    img = np.array(img)
    img = img / 255.0
    return img.astype(np.float32)

def mse(imageA, imageB):
    if len(imageA.shape) > 2 and imageA.shape[2] == 3:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    if len(imageB.shape) > 2 and imageB.shape[2] == 3:
        imageB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
        
    # Compute the mean squared error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
