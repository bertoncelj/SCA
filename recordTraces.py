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
            writeSerial(x + "\0")

        print(value)
        find = value.find(b'decrypted:' + str.encode(x)) # printing the value
        print(find)

def generateRandomData():
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))

def generateRandomNum():
    return "".join([str(random.randint(0,9)) for _ in range(16)])

def generateRandomNumInt():
    aa = lambda x: (chr(x), hex(x))
    bb = lambda x: (str(x[0]) + str(x[1]))
    randomIntNums = [aa(random.randint(32,127)) for _ in range(16)]
    stringKey = "".join([x[0] for x in randomIntNums])
    hexKey = [x[1] for x in randomIntNums]
    hexKeyStrWith0x = "".join(hexKey)

    listKeyHex = list(zip(hexKeyStrWith0x[::2],hexKeyStrWith0x[1::2]))[1::2]
    # strKeyHex = "".join(listKeyHex)
    strKeyHex = "".join(list(map(bb,listKeyHex)))

    print("hexKeyStr:" , strKeyHex)
    print("stringKey:" , stringKey)

    return strKeyHex, stringKey

#Save protocol
start_file_menu  =(16, 40)
save_as = (44, 150)
# outofocus on title
# change title keyboard
# change type
select_type_menu = (954,661)
cvs_type = (969, 723)
save_button = (1010, 700)


start_trigger = (87, 782)
write_cursor = (359, 133)

# while True:
#     num = generateRandomNum()
#     writeSerial(num + "\0")
#     readSerial(num)
#     arduino.flush()

# num = "0123456789012345"

# while True:
#     hexKeyStr, ASCIIkey = generateRandomNumInt()
#     sleep(1)
#     writeSerial(ASCIIkey + "\0")
#     readSerial(ASCIIkey)
#     arduino.flush()

# while True:
    # print(mouse.get_position())
count=0
while True:
    print("trace num: ", count)
    count = count+1

    mouse.click('left')

    # start
    reset_cursor()

    print(mouse.get_position())

    hexKeyStr, ASCIIkey = generateRandomNumInt()
    # num = "0123456789012345"
    clickMouse(*start_trigger)
    print("SEND")
    writeSerial(ASCIIkey + "\0")
    readSerial(ASCIIkey)
    arduino.flush()

    time.sleep(.5)
    clickMouse(*start_file_menu)
    clickMouse(*save_as)
    clickMouse(*write_cursor)

# save by data name
    stringToType = "aes_forward_" + hexKeyStr + "_"
    pyautogui.typewrite(stringToType)

# change data type
    clickMouse(*select_type_menu)
    clickMouse(*cvs_type)
    clickMouse(*save_button)

