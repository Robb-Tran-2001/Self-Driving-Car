import cv2
from random import randrange

#Video source
video = cv2.VideoCapture('dashcam.mp4')
#Create car and pedestrian classifier from trained data
car_tracker = cv2.CascadeClassifier('cars.xml')
people_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    #Read each frame
    (success, frame) = video.read()

    #If can't read skip to next frame
    if success==False:
        continue

    #Convert to grayscale
    frame_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Get the coords of cars and pedestrians
    car_coords = car_tracker.detectMultiScale(frame_gs)
    people_coords = people_tracker.detectMultiScale(frame_gs)

    #Draw rectangle shapes over them
    for (x, y, w, h) in car_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 256), 2)
    for (x, y, w, h) in people_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 256, 0), 2)

    #Render
    cv2.imshow('Life Footage', frame)
    key = cv2.waitKey(1)

    #End if q or Q is pressed
    if key == 81 or key == 113:
        break

#Release
video.release()

print("Code Completed")