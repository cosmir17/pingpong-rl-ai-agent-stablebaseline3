import time
import gymnasium
from gymnasium import spaces
from nq_screen_extractor import *
import pyautogui
from tensorflow import keras
import matplotlib.pyplot as plt
from score_activity_extractor import detect_minus_sign
from tensorflow.keras.preprocessing.image import save_img

class NQEnv(gymnasium.Env):
    metadata = {'render.modes': ['console']}
    actions = ['0th', '1st', '2nd', '3rd', '4th', '5th', 'nothing']
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 40, 40), dtype=np.uint8)
        self.take_screenshot_save_to_selfstate()
        self.minus_sign_presence = False
        self.ball_not_shown_between_peddles_iteration = 0 
        pyautogui.moveTo(1300, 300)          
        self.click()
        
    def reset(self, seed=None): #reset this python app's state for a new game
        print("===========================================")
        print("RESET is called")
        pyautogui.moveTo(1300, 300)          
        self.click()
        self.take_screenshot_save_to_selfstate()
        self.ball_not_shown_between_peddles_iteration = 0 
        self.minus_sign_presence = False
        return self._state, {}

    def step(self, action): #the neural network decides an action based on previous turn's state
        try:
            str(self.actions[action])
        except:
            print("only integer scalar array error: " + str(action))
            screenshot = self.take_screenshot_save_to_selfstate()
            self.minus_sign_presence = detect_minus_sign(screenshot)
            return self._state, 0.0, False, False, {}

        self.press_key(action)
        scored, screenshot = self.take_screenshot_is_scored()

        if self._stage == GameStage.inifite_pong_screen:
            print("Episode is done")
            # if scored:
            #     save_img('game_ended_but_minus_sign.png', screenshot)
            reward = -1.0
            print("action: " + str(self.actions[action]) + "  which stage: " + str(self._stage) + " reward: " + str(
                reward))
            return self._state, reward, True, False, {}
        
        if self._stage == GameStage.stuck_no_ball_present_during_gameplay and self.ball_not_shown_between_peddles_iteration > 15:
            print("Stuck state - reward : -0.3")
            reward = -0.3
            pyautogui.press('r')
            time.sleep(0.05)
            return self._state, reward, True, False, {}
            
        if self._stage == GameStage.stuck_no_ball_present_during_gameplay:
            self.ball_not_shown_between_peddles_iteration += 1 
            reward = 0.02
            print("Possible Stuck state => action: " + str(self.actions[action]) + "  which stage: " + str(self._stage) + " reward: " + str(
                reward))
            return self._state, reward, False, False, {}
            
        if scored:
            reward = 0.2
            print("action: " + str(self.actions[action]) + "  which stage: " + str(self._stage) + " reward: " + str(
                reward))
            return self._state, reward, False, False, {}
    
        else:
            reward = 0.02
            print("action: " + str(self.actions[action]) + "  which stage: " + str(self._stage) + " reward: " + str(
                reward))
            return self._state, reward, False, False, {}
    
    def press_key(self, action):
        starting_position = 90
        interval = 70        
        _, y = pyautogui.position()

        if action == 0:
           pyautogui.moveTo(1300, starting_position)
        if action == 1:
           pyautogui.moveTo(1300, starting_position + (interval * 1))
        if action == 2:
           pyautogui.moveTo(1300, starting_position + (interval * 2))
        if action == 3:
           pyautogui.moveTo(1300, starting_position + (interval * 3))
        if action == 4:
           pyautogui.moveTo(1300, starting_position + (interval * 4))
        if action == 5:
           pyautogui.moveTo(1300, starting_position + (interval * 5))
        if action == 6:
            time.sleep(0.0001)
            return 0
        else:
            time.sleep(0.0001)
            return 0
            
    def take_screenshot_is_scored(self):
        screenshot = self.take_screenshot_save_to_selfstate()
        previous_round_minus_sign_presence = self.minus_sign_presence
        this_round_minus_sign_presence = detect_minus_sign(screenshot) 
        self.minus_sign_presence = this_round_minus_sign_presence
        scored = previous_round_minus_sign_presence != this_round_minus_sign_presence
        return scored, screenshot

    def take_screenshot_save_to_selfstate(self):
        screenshot = capture_window()
        stage_enum = which_stage(screenshot)
        img_processed = tf.image.resize(screenshot, (40, 40))
        img_processed = tf.transpose(img_processed, (2, 0, 1))
        # img_processed_squeezed = tf.squeeze(img_processed, axis=0)
        # plt.imshow(img_processed_squeezed, cmap='gray')
        # plt.show()
        # img_processed = keras.preprocessing.image.img_to_array(img_processed)
        img_processed = keras.preprocessing.image.img_to_array(img_processed)
        img_processed = img_processed.astype(np.uint8)
        # img_processed = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        # img_processed = cv2.resize(img_processed, (100, 60), interpolation=cv2.INTER_AREA)
        # img_processed = np.expand_dims(img_processed, axis=-1)
        self._state = img_processed
        self._stage = stage_enum
        return screenshot

    def click(self):
        time.sleep(0.05)
        pyautogui.click()
        time.sleep(0.005)


    # def press_key_return_penalty(self, action):
    #     upper_line = 10
    #     lower_line = 500
    #     _, y = pyautogui.position()

    #     if action == 0 : #up
    #         if y > upper_line:
    #             pyautogui.move(0, -70)
    #             return 0
    #         else:
    #             return (upper_line - y) * -0.02

    #     if action == 1: #down
    #         if y < lower_line:
    #             pyautogui.move(0, 70)
    #             return 0
    #         else:
    #             return (y - lower_line) * -0.02

    #     if action == 2:
    #         time.sleep(0.0001)
    #         return 0
    #     else:
    #         time.sleep(0.0001)
    #         return 0