def detect_minus_sign(game_img):
    x, y = 550, 8
    intensity = game_img[y, x, 0].numpy()
    if (intensity == 128):
        return True
    else:
        return False
