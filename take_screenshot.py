import numpy as np
import pyautogui
import imutils
import cv2

i = 1
image = pyautogui.screenshot()
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
resizedimage = cv2.resize(image,(256,256))
cv2.imwrite("/home/labuser/DeepLearning/screenshots/"+str(i)+".png", resizedimage)
i = i + 1
