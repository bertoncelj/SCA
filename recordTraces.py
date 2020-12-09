import mouse
# Importing Libraries
import serial
import time
import random
import string

import keyboard
import pyautogui, sys
from time import sleep

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)

def reset_cursor():
    mouse.move(-10000,-10000, absolute=False, duration=0.0)

def clickMouse(x,y):
    reset_cursor()
    mouse.move(x,y, absolute=False, duration=0.0)

    sleep(0.1)
    # mouse.click('left')
    pyautogui.click(button='left')
    print(mouse.get_position())
    sleep(.333)

def writeSerial(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(.008)

def readSerial(x):
    value=b''
    find=0
    while find <= 0:
        value += arduino.read(300)
        # resend
        if value == b'':
            writeSerial(x)

        print(value)
        find = value.find(b'decrypted:' + str.encode(num)) # printing the value
        print(find)

def generateRandomData():
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))

def generateRandomNum():
    return "".join([str(random.randint(0,9)) for _ in range(16)])

#Save protocol
start_file_menu  =(16, 40)
save_as = (44, 150)
# outofocus on title
# change title keyboard
# change type
select_type_menu = (954,714)
cvs_type = (969, 769)
save_button = (1100, 758)


start_trigger = (87, 782)
write_cursor = (269, 68)

# while True:
    # print(mouse.get_position())
    # num = generateRandomNum()
    # stringToType = "aes_forward_" + num
    # keyboard.write(stringToType)
    # time.sleep(2)

while True:
    mouse.click('left')

    # start
    reset_cursor()

    print(mouse.get_position())

    num = generateRandomNum()
    # num = "0123456789012345"
    clickMouse(*start_trigger)
    print("SEND")
    writeSerial(num + "\0")
    readSerial(num + "\0")
    arduino.flush()

    time.sleep(.5)
    clickMouse(*start_file_menu)
    clickMouse(*save_as)
    clickMouse(*write_cursor)

# save by data name
    stringToType = "aes_forward_" + num + "_"
    pyautogui.typewrite(stringToType)

# change data type
    clickMouse(*select_type_menu)
    clickMouse(*cvs_type)
    clickMouse(*save_button)

    print("next")

