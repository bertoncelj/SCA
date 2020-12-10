import os
import numpy as np
import csv

def findAllUnderScores(FILE_PATH, num, listR):
    num = FILE_PATH.find("_",num)
    if num == -1:
        return listR
    else:
        listR.append(num)
        return findAllUnderScores(FILE_PATH, num+1, listR)

def getDataFromFileName(FILE_PATH):
    cpArray = findAllUnderScores(FILE_PATH, 0, [])
    dataStr = str(FILE_PATH[cpArray[1]+1:cpArray[2]])
    print(dataStr)
    tupData = (list(zip(dataStr[::2], dataStr[1::2])))
    nn = [x[0]+x[1] for x in tupData]
    stToHex = lambda x: hex(int(x,16))
    stToInt = lambda x: int(x,16)
    dataHex = list(map(stToHex , nn))
    keyInt = list(map(stToInt , nn))
    dataInt = np.array(keyInt, dtype="uint8")

    print("data hex: ", dataHex)
    print("data int: ", dataInt)
    # return keyHex
    return dataInt

def readFile(path):
    with open('traTest/' + path, newline='') as csvfile:
        spamreader = iter(csv.reader(csvfile, delimiter=' ', quotechar='|'))
        trsList = []
        for idx, row in enumerate(spamreader):
            if idx > 2:
                strWholeData = ', '.join(row)
                splited = strWholeData.split(',')
                data = float(splited[1])
                trigger = float(splited[2])
                if trigger > 1:
                    trsList.append(data)
        return trsList

PATH = "/home/tine/Documents/UNI/FRI/Kriptologija/project/SCA/traTest"
directories = os.listdir(PATH)
traces = np.empty(shape=(10, 8770), dtype='float32')
data = np.empty(shape=(10, 8), dtype='uint8')

for idx, file in enumerate(directories):
    if idx == 10:
        break

    print(file)
    trace = readFile(file)
    traces[idx, :] = np.array(trace[:8770], dtype = "float32")
    data[idx, :] = np.array(getDataFromFileName(file), dtype = "uint8")
    print("len trace: ", len(trace))



# print("Saving file")
# np.savez("testNpz", traces=traces, data=data)
# print("Done")





