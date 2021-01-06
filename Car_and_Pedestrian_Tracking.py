import cv2
from random import randrange

#Image source
street = 'street.jpg'

#Create car and pedestrian classifier from trained data
car_tracker = cv2.CascadeClassifier('cars.xml')
people_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

#Create opencv Image
street_img = cv2.imread(street)

#Convert to black and white
street_img_gs = cv2.cvtColor(street_img, cv2.COLOR_BGR2GRAY)

#Detect the car coords and people coords
car_coords = car_tracker.detectMultiScale(street_img_gs)
people_coords = people_tracker.detectMultiScale(street_img_gs)
print(car_coords)
print(people_coords)

#Draw rectangles over each car and pedestrian
for (x, y, w, h) in car_coords:
    cv2.rectangle(street_img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
for (x, y, w, h) in people_coords:
    cv2.rectangle(street_img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)


#Display image with rectangles drawn over car and people
cv2.imshow('Car and Pedestrian Image', street_img)

#Necessary for display
cv2.waitKey()

print("Code Completed")