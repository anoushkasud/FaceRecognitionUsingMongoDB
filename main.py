import cv2
import face_recognition
import numpy as np
from datetime import datetime
from pymongo import MongoClient

connection = 'mongodb://localhost:27017/'
client = MongoClient(connection)
database = 'AttendanceSystem'; collection = 'deepface'; collection2 = 'attendances'
db = client[database]

#webcam
cap = cv2.VideoCapture(0)
cap.set(3,640)#width
cap.set(4,480)#height

#load the encoding file
studentIds = []
embeddings = []
for x in db[collection].find():
    studentIds.append(x['studentId'])
    embeddings.append(x['embedding'])

modeType = 0
counter = 0
id = -1
while True:
    success, img = cap.read()

    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    if faceCurFrame:
        for encodeFace in encodeCurFrame:
            matches = face_recognition.compare_faces(embeddings,encodeFace)#
            faceDist = face_recognition.face_distance(embeddings,encodeFace)#
            matchIndex = np.argmin(faceDist)

            if matches[matchIndex]:
                # print("face detected")
                id = studentIds[matchIndex]#

                if counter == 0:
                    counter = 1
                    modeType = 1
                    print("uploading to database");

        if counter !=0 :
            if counter ==1:
                #get the data
                myquery = { "courseId": "44GH12Y7", "studentId":id }#####
                last_doc = db[collection2].find_one(myquery,sort=[('attendanceDate', -1)])
                last_date = last_doc['attendanceDate']
                # print(last_doc['attendanceDate'])
 
                #update data of attendance
                today = datetime.now()
                # print((today- last_date).days)

                daysElapsed = (today- last_date).days
                # put into db
                if daysElapsed >0:
                    x = db[collection2].insert_one({ "studentId": id, "courseId": "44GH12Y7", 'attendanceDate': today })
                else:
                    modeType = 3
                    print("already marked")
                    counter = 0

            if modeType !=3:
                if 10<counter<20:
                    modeType = 2
                    print("marked")
                
                counter +=1   

                if counter>=20:
                    counter = 0
                    modeType = 0
                    print('active')
    else:
        modeType = 0
        print('active')
        counter = 0            

    cv2.waitKey(2)