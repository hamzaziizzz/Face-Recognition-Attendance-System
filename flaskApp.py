'''A minimalistic flask app for Yolov7'''
from flask import Flask, render_template, Response, jsonify, request, session
from flask_bootstrap import Bootstrap
from deepface_video import *

app = Flask(__name__)
Bootstrap(app)

app.config['SECRET_KEY'] = 'grilsessionkey'

video_capture = capture_video()
output = face_recognition(video_capture, "DATABASE")

@app.route("/",methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])
def home():
    return render_template('index.html')

def generate_frames():
    for success, frame in output:
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route("/activateWebCam")
def activateWebCam():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# def generate_frames(path_x = '',conf_= 0.25):
#     yolo_output = video_detection(path_x,conf_)
#     for detection_,FPS_,xl,yl in yolo_output:
#         ref,buffer=cv2.imencode('.jpg',detection_)
#         frame=buffer.tobytes()
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

# @app.route('/FrontPage')
# @app.route('/video')
# def video():
#     return Response(generate_frames(path_x = 'static/files/vid.mp4',conf_=0.75),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
