from pynput import keyboard
import pyautogui as pag
from pynput.mouse import Controller
import time as tm
def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        if key.char=='`':
            tm.sleep(0.5)
            mouse = Controller()
            cur_x, cur_y = pag.position()
            pag.click(button='middle')
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False
# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

'''
import numpy as np
import pyautogui as pag
from pynput.keyboard import Key, Listener
from pynput.mouse import Controller
def on_press(key):
    print('{0} pressed'.format(
        key))
def on_click(x, y, button, pressed):
            if not pressed:
                return False
def on_release(key):
    print('{0} release'.format(
        key))
    if key.char=='`':
        mouse = Controller()
        cur_x, cur_y = pag.position()
        pag.click(x=cur_x, y=cur_y, button='middle')
    if key == Key.esc:
        # Stop listener
        return False
    
# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()


from pynput.mouse import Controller
mouse = Controller()
def on_click(x, y, button, pressed):
    print(f"{'按下' if pressed else '釋放'} ，當前位置是： {(x, y)}")
    if not pressed:
        return False
cur_x, cur_y = pag.position()
pag.click(x=cur_x, y=cur_y, button='middle')

'''