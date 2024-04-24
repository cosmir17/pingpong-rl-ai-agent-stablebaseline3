# import cv2
import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import load_img

# image_file_path = "game_inprogress_scored.png"
# image_file_path = "23.png"
# game_img = load_img(image_file_path)

from PIL import Image

def detect_minus_sign(game_img):
    x, y = 550, 8
    intensity = game_img[y, x, 0].numpy()
    if (intensity == 128):
        return True
    else:
        return False    

# r, g, b, a = game_img.getpixel((x, y))
# print(color)
# print(r, g, b, a)
# plt.imshow(game_img, cmap='gray')
# plt.show()

    # x, y = 550, 8
    # game_img = Image.fromarray(np.uint8(game_img.numpy()))
    # # color = game_img.getpixel((x, y))
    # color = game_img[y, x].numpy()
    # if (color == (128, 128, 128)):
    #     return True
    # else:
    #     return False    