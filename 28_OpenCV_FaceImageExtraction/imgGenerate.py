import numpy as np
import cv2 as cv2
import sys

#im = cv2.imread("ManMohan\frame1.jpg")

#cv2.imshow('Videoc', im)
#cascPath = sys.argv[1] 

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
count = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
        #maxSize=(150,150)
        #,
        #flags=cv2.CASCADE_SCALE_IMAGE
    )
    
   
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        count = count+1
        x0,y0=int(x),int(y)
        x1,y1=int(x+w),int(y+h)
        roi=frame[y0:y1,x0:x1]#crop 
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
       
        cropped=cv2.resize(roi, dsize=(150,150))
        cv2.imshow('Videoc', cropped)
        cv2.imwrite("output/frame%d.jpg" % count, cropped)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()