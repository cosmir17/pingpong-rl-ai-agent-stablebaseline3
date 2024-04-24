import pytesseract
import cv2
import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import save_img

image_file_path = "game_inprogress.png"
import numpy as np

# Load the image
img = cv2.imread(image_file_path)
# img_np = np.array(img)


# Convert the image to a NumPy array


# Display the thresholded image
# plt.imshow(img_adaptive, cmap='gray')
# plt.show()

# Identify the location of the score on the screen (you'll need to adjust these values)
score_region_x = img.shape[1] // 2 - 50  # center of the image, minus an offset to get the start of the score region
score_region_y = 98  # some pixels from the top, you may need to adjust this
score_region_width = 100  # estimated width of the score region, you may need to adjust this
score_region_height = 50  # estimated height of the score region, you may need to adjust this

# Extract the score region from the screenshot

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

score_region = img[score_region_y:score_region_y+score_region_height, score_region_x:score_region_x+score_region_width]


# img_adaptive = img_gray
# score_region = cv2.adaptiveThreshold(score_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

_, score_img_binary = cv2.threshold(score_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert the colors in the image
score_img_inverted = cv2.bitwise_not(score_img_binary)

blurred_img = cv2.GaussianBlur(score_img_inverted, (5, 5), 0)
_, binarized_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


plt.imshow(binarized_img, cmap='gray')
plt.show()

# cv2.imwrite('binarized_img.png', score_region)

# Apply OCR to the score region
# score = pytesseract.image_to_string(binarized_img, config='--oem 3 --psm 6 outputbase digits')
score = pytesseract.image_to_string(binarized_img, config='--oem 3 --psm 10 outputbase digits')
# score = pytesseract.image_to_string(score_img_inverted, config='--oem 3 --psm 6')


print("score is " + score)