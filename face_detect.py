import cv2
import os
import sys
import csv

directory = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

csv_output = csv.writer(open("output.csv", "w"))
csv_output.writerow(["filename", "num_faces"])

faceCascade = cv2.CascadeClassifier(cascPath)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for filename in os.listdir(directory):
    temp = []
    if filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(directory, filename))
        image2 = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #clahe_img = clahe.apply(gray) 
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(200, 200))

        csv_output.writerow([filename, len(faces)])
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            sub_face = image[y:y + h, x:x + w]
            face_file_name = "faces/face_" + str(y) + ".jpg"
            cv2.imwrite(face_file_name, sub_face)
        #cv2.imshow("Faces found", image2)
        #cv2.waitKey(0)