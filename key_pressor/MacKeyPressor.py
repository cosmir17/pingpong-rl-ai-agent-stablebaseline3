from key_pressor.AbstractKeyPressor import AbstractKeyPressor
import time
import os

class MacKeyPressor(AbstractKeyPressor):

    @property
    def mouse_click(self):
        print("mouse click")
        return 34

    @property
    def mouse_up(self):
        print("mouse up")
        return 28

    @property
    def mouse_down(self):
        print("mouse down")
        return 40

    def press_key(self, key):
        cmd = "osascript -e 'tell application \"System Events\" to key code \"" + str(key) + "\"'"
        os.system(cmd)
        time.sleep(0.01)
