#A python module to capture a video stream and perform facial recognition on it.

import threading #A thread is a separate flow of execution
import time
import cv2
from deepface import DeepFace
from deepface.detectors import FaceDetector
import numpy as np
import time
import pandas as pd

attendence_list = set() # A set is defined to store the  
start_time=time.time() # Starting time of the function to execute attendence only for a perticular time period 

#function to extract the name from the returned path.

def format_name(name: str) -> str:
    '''Format name to be displayed on video frame
    '''
    name = name.split('\\')[-1].split('/')[0]
    attendence_list.add(name)
    return name

# capture video using cv2 and set the video frames to 5.

def capture_video():
    '''Capture a video stream and return it.
    '''
    # Capture video from webcam
    video_capture = cv2.VideoCapture(0)
    
    #reduce fps
    video_capture.set(cv2.CAP_PROP_FPS, 5)
    return video_capture


def store_unknown_faces(frame, current_time):
    '''Store unknown faces in a folder.
    '''
    #save current frame as unknown face
    cv2.imwrite(f'unknown_faces/{current_time}.jpg', frame)

def face_recognition(video_capture, db_path):
    '''Perform facial recognition on a video stream.
    '''
    while True:
        
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Perform facial recognition
        detector_name = "mtcnn" # "opencv" or "ssd"

        detector = FaceDetector.build_model(detector_name) #set opencv, ssd, dlib, mtcnn or retinaface

        obj = FaceDetector.detect_faces(detector, detector_name, frame, align=True)  #return the path of matched images

        face_recognized = ''
        dy = 0
        
        if len(obj) > 0:
            #Create empty numpy array
            imgs = [obj[0][0]]

            for i in range(1, len(obj)):
                imgs.append(obj[i][0])
            
            #print(imgs)

            # Perform facial recognition and finding path for images
            df = DeepFace.find(imgs, db_path = db_path, silent=True, enforce_detection=False, prog_bar=False)

        
        for i in range(len(obj)):
            # Do facial recognition on each face
            #df = DeepFace.find(img_path = obj[i][0], db_path = "train/", silent=True, enforce_detection=False, prog_bar=False)
            
            #current time in hours, minutes and seconds
            current_time = time.strftime("%H:%M:%S", time.localtime())
            #check if df is list
            if isinstance(df, list):
                #get the name of the person
                face_recognized = df[i].iloc[0]['identity']
                face_recognition_distance = df[i].iloc[0]['VGG-Face_cosine']
            else:
                face_recognized = df.iloc[0]['identity']
                face_recognition_distance = df.iloc[0]['VGG-Face_cosine']
            face_recognized = format_name(face_recognized)+f' Similarity_Distance:{face_recognition_distance:.3f}'
            if face_recognition_distance > 0.2:
                face_recognized = 'Unknown '+current_time
                #create a thread for storing unknown faces
                #Uncomment the following line to store unknown faces
                threading.Thread(target=store_unknown_faces, args=(frame.copy(),current_time)).start()
                #set bounding box color to red
                color = (0, 0, 255)
            else:
                #set bounding box color to green
                color = (0, 255, 0)

            
            #draw rectangle on face
            #Reference - https://github.com/serengil/deepface/blob/master/deepface/commons/realtime.py
            x = obj[i][1][0]; y = obj[i][1][1]
            w = obj[i][1][2]; h = obj[i][1][3]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2) #draw rectangle to main image
            #Display time at top middle of screen
            cv2.putText(frame, current_time, (int(frame.shape[1]/2)-30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, face_recognized, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            dy+=30
            
            
        
        # Display the resulting frame
        # cv2.imshow('Video', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

        # stop recording attendence after 1 minute
        present_time=time.time()
        if present_time - start_time > 60:
            break
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# def main():
#     '''
#     Main function.
#     '''
#     # Capture video stream
#     video_capture = capture_video()

#     # Perform facial recognition
#     face_recognition(video_capture, "DATABASE"))

#     # Release video capture
#     video_capture.release()

df = pd.DataFrame(attendence_list) # Converting attendence list to pandas dataframe.
df.to_csv(".\\Attendance-CSV\\attendance.csv", index=False) # saving the attendence in the form of csv file.




# if __name__ == "__main__":
#     main()
#     df = pd.DataFrame(attendence_list) # Converting attendence list to pandas dataframe.
#     df.to_csv(".\\Attendance-CSV\\attendance.csv", index=False) # saving the attendence in the form of csv file.