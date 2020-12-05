import mouse

# Importing Libraries
import serial
import time
import random
import string

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)

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

while True:
    num = generateRandomNum()
    print("SEND")
    writeSerial(num + "\0")
    readSerial(num + "\0")
    arduino.flush()
    time.sleep(.005)


# import keyboard
# import pyautogui

# from time import sleep

# firefox = (33, 58)
# obs = (31, 826)
# play = (86, 1061)
# vu = (156, 68)
# rp = (392, 72)
# present = (154, 332)
# video_duration = 60
# Crp = (500,75)


# mouse.click('left')

# def reset_cursor():
#     mouse.move(-10000,-10000, absolute=False, duration=0.0)

# def move_cursor(x,y):
#     reset_cursor()
#     mouse.move(x,y, absolute=False, duration=0.0)

#     sleep(0.1)
#     mouse.click('right')
#     print(mouse.get_position())
#     sleep(1)

# while(True):
    # print(mouse.get_position())

#start
#reset_cursor()

#move_cursor(*firefox)

##loop start
#move_cursor(*vu)


##presentation
#move_cursor(154, 332)

##recording window
#move_cursor(*rp)
#sleep(10)


##playing
#keyboard.write('s')
#move_cursor(*play)
#sleep(video_duration)
##stop resording
#keyboard.write('s')
#move_cursor(*Crp)
##loop close
