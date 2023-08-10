
import calendar
import csv
import os
from flask import Flask, render_template, Response, jsonify,request
from camera import VideoCamera
import cv2
from camera import getImagesAndLabels
import numpy as np
import pandas as pd
import datetime
import time

app = Flask(__name__)


ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d %m %Y')
day = datetime.datetime.strptime(date, '%d %m %Y').weekday()
day_names=calendar.day_name[day]
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
def data_IT():
    row = [Enrollment, name, Email, Department, semester, Division]
    col_names = ['Enrollment', 'name', 'Email', 'Department', 'semester', 'Division']
    exists1 = os.path.isfile('StudentDetails/IT\St_Details_IT_' + semester + '_' + Division + '.csv')
    if Division == "A":

        if exists1:
            try:
                with open('StudentDetails/IT\St_Details_IT_' + semester + '_' + Division + '.csv',
                          newline='') as csvFile:
                    #     csv_reader = csv.reader(csvFile)
                    df = pd.read_csv(csvFile)
                    if str(Enrollment) in df['Enrollment'].to_string(index=False):
                        return render_template("registered.html")
                    else:
                        with open('StudentDetails/IT\St_Details_IT_' + semester + '_' + Division + '.csv',
                                  'a+') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerow(row)
                        csvFile.close()
                        return render_template('frame_data.html',f_nm=name,f_enroll=Enrollment)
            except IndexError:
                return render_template("error.html")
        else:
            with open('StudentDetails/IT\St_Details_IT_' + semester + '_' + Division + '.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(col_names)
                writer.writerow(row)
            csvFile.close()
            return render_template('frame_data.html',f_nm=name,f_enroll=Enrollment)
    elif Division == "B":
        if exists1:
            try:
                with open('StudentDetails/IT\St_Details_IT_' + semester + '_' + Division + '.csv',
                          newline='') as csvFile:
                    #     csv_reader = csv.reader(csvFile)
                    df = pd.read_csv(csvFile)
                    if str(Enrollment) in df['Enrollment'].to_string(index=False):
                        return render_template("registered.html")
                with open('StudentDetails/IT\St_Details_IT_' + semester + '_' + Division + '.csv', 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()
                return render_template('frame_data.html',f_nm=name,f_enroll=Enrollment)
            except IndexError:
                return render_template("error.html")
        else:
            with open('StudentDetails/IT\St_Details_IT_' + semester + '_' + Division + '.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(col_names)
                writer.writerow(row)
            csvFile.close()
            return render_template('frame_data.html',f_nm=name,f_enroll=Enrollment)


def data_CE():
    row = get_image.row
    col_names = ['Enrollment', 'name', 'Email', 'Department', 'semester', 'Division']
    exists1 = os.path.isfile('StudentDetails/Computer\St_Details_CE_' + semester + '_' + Division + '.csv')
    if Division == "A":
        if exists1:
            try:
                with open('StudentDetails/Computer\St_Details_CE_' + semester + '_' + Division + '.csv', newline='') as csvFile:
                            #     csv_reader = csv.reader(csvFile)
                    df = pd.read_csv(csvFile)
                    if str(Enrollment) in df['Enrollment'].to_string(index=False):
                        return render_template("registered.html")
                    else:
                        with open('StudentDetails/Computer\St_Details_CE_' + semester + '_' + Division + '.csv', 'a+') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerow(row)
                        csvFile.close()
                        return render_template('frame_data.html',f_nm=name,f_enroll=Enrollment)
            except IndexError:
                return render_template("error.html")
        else:
            with open('StudentDetails/Computer\St_Details_CE_' + semester + '_' + Division + '.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(col_names)
                writer.writerow(row)
            csvFile.close()
            return render_template('frame_data.html',f_nm=name,f_enroll=Enrollment)
    elif Division == "B":
        if exists1:
            try:
                with open('StudentDetails/Computer\St_Details_CE_' + semester + '_' + Division + '.csv', newline='') as csvFile:
                            #     csv_reader = csv.reader(csvFile)
                    df = pd.read_csv(csvFile)
                    if str(Enrollment) in df['Enrollment'].to_string(index=False):
                        return render_template("registered.html")
                with open('StudentDetails/Computer\St_Details_CE_' + semester + '_' + Division + '.csv', 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                    csvFile.close()
                return render_template('frame_data.html',f_nm=name,f_enroll=Enrollment)
            except IndexError:
                return render_template("error.html")
        else:
            with open('StudentDetails/Computer\St_Details_CE_' + semester + '_' + Division + '.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(col_names)
                writer.writerow(row)
            csvFile.close()
            return render_template('frame_data.html',f_nm=name,f_enroll=Enrollment)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get_image', methods=["POST","GET"] )
def get_image():
    global name, Enrollment, Email, Department, semester, Division
    if request.method == "POST":
        name=request.form["st"]
        Enrollment = request.form["en"]
        Email = request.form["email"]
        Department = request.form["dept"]
        semester = request.form["sem"]
        Division = request.form["div"]
        get_image.row = [Enrollment, name, Email, Department, semester, Division]
        # col_names=['Enrollment', 'name', 'Email', 'Department', 'semester', 'Division']

        if Department=="Computer":
            if semester=="1":
                return data_CE()
            if semester=="2":
                return data_CE()
            if semester=="3":
                return data_CE()
            if semester=="4":
                return data_CE()
            if semester=="5":
                return data_CE()
            if semester=="6":
                return data_CE()
            if semester=="7":
                return data_CE()
            if semester=="8":
               return data_CE()
        if Department=="Information Technology":
            if semester=="1":
                return data_IT()
            if semester=="2":
                return data_IT()
            if semester=="3":
                return data_IT()
            if semester=="4":
                return data_IT()
            if semester=="5":
                return data_IT()
            if semester=="6":
                return data_IT()
            if semester=="7":
                return data_IT()
            if semester=="8":
                return data_IT()

@app.route('/train_model' )
def train_model():
    return render_template('train_m.html')

@app.route('/detection')
def index():
    if day_names=="Saturday" or day_names=="Sunday":
        return render_template("holiday.html")
    else:
        return render_template('detection_interface.html')

@app.route('/training')
def training():
    parent_dir = "TrainingImages"
    directory = str(Department) + "\Semester_" + str(semester) + "\DIV_" + str(Division)
    path1 = os.path.join(parent_dir, directory)
    cv2.destroyAllWindows()
    # recognizer = cv2.face_LBPHFaceRecognizer.create()#cv2.face.LBPHFaceRecognizer_create()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels(path1)
    except Exception as e:
        print(e)
    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("TrainingImageLabel/"+str(Department) + "/Semester_" + str(semester) + "/DIV_" + str(Division)+"/Trainner.yml")
    except Exception as e:
        print(e)
    # recognizer = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    # harcascadePath = "haarcascade_frontalface_default.xml"
    # detector = cv2.CascadeClassifier(harcascadePath)
    # faces, Id = getImagesAndLabels("TrainingImages")
    # recognizer.train(faces, np.array(Id))
    # recognizer.save("TrainingImageLabel\Trainner.yml")
    return render_template('training.html')

@app.route('/start_camera')
def Take_images():
    exists_dir = os.path.isdir("TrainingImages\ " + str(Department) + "\Semester_" + str(semester) + "\ " + str(Division))
    if not exists_dir:
        parent_dir = "TrainingImages"
        directory = str(Department) + "\Semester_" + str(semester) + "\DIV_" + str(Division)
        path1 = os.path.join(parent_dir, directory)
        os.makedirs(path1, exist_ok=True)
        print(path1)
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    #incrementing sample number
                sampleNum = sampleNum + 1
                # cv2.imwrite("TrainingImages\ "+ name + "." + Enrollment + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                #         #display the frame
                # cv2.imwrite("TrainingImages\ " + str(Department) + "\Semester_" + str(semester) + "\ " + name + "." + Enrollment + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # cv2.imwrite("TrainingImages\Computer\Semester_1\DIV_A\ "+ name + "." + Enrollment + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imwrite(os.path.join(path1,name + "." + Enrollment + '.' + str(sampleNum) + ".jpg"),gray[y:y + h, x:x + w])
                cv2.imshow('frame',img)
                ret, jpeg = cv2.imencode('.jpg', img)
                # wait for 100 miliseconds
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                # break if the sample number is morethan 100
            elif sampleNum>20:
                break
    elif exists_dir:
        parent_dir = "TrainingImages"
        directory = str(Department) + "\Semester_" + str(semester) + "\DIV_" + str(Division)
        path1 = os.path.join(parent_dir, directory)
        # os.makedirs(path1, exist_ok=True)
        print(path1)
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # cv2.imwrite("TrainingImages\ "+ name + "." + Enrollment + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                #         #display the frame
                # cv2.imwrite("TrainingImages\ " + str(Department) + "\Semester_" + str(semester) + "\ " + name + "." + Enrollment + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # cv2.imwrite("TrainingImages\Computer\Semester_1\DIV_A\ "+ name + "." + Enrollment + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imwrite(os.path.join(path1, name + "." + Enrollment + '.' + str(sampleNum) + ".jpg"),
                            gray[y:y + h, x:x + w])
                cv2.imshow('frame', img)
                ret, jpeg = cv2.imencode('.jpg', img)
                # wait for 100 miliseconds
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                # break if the sample number is morethan 100
            elif sampleNum > 20:
                break
    cam.release()
    cv2.destroyAllWindows()
    return render_template('frame_data.html', f_nm=name, f_enroll=Enrollment)

@app.route('/get_attendance', methods=['POST'])
def get_attendance():
    global semester_a,department_a,division_a,subject
    if request.method == "POST":
        semester_a=request.form["sem0"]
        department_a=request.form["dept0"]
        division_a=request.form["div0"]
        if department_a=="Computer":
            if semester_a == "1":
                if division_a=="A":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-1"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ES"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-1_Tutorial"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEE"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Physics"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EEE_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Physics"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ES"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Physics_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-1"
                            return attendance_take_CE()
                        if timeStamp >= '23:00:00' and timeStamp <= '23:59:00':
                            subject="EG"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EG_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEE"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CS"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                if division_a=="B":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EG_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEE"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CS"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Physics_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-1"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="EG"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-1_Tutorial"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEE"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Physics"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EEE_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Physics"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ES"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-1"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ES"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
            if semester_a == "2":
                if division_a=="A":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CPD_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="CPU"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="BE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CPU_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-2"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CPD"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-2_Tutorial"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EME"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="BE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="BE_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EME"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CPU"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EME_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="CPU"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Maths-2"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                if division_a=="B":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EME_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="CPU"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Maths-2"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="BE_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EME"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CPU"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CPU"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-2"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CPD"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CPD_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="CPU"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="BE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-2_Tutorial"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EME"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="BE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
            if semester_a == "3":
                if division_a=="A":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="DBMS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DBMS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="DS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Maths-3"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DE_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-3"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DS"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-3_Tutorial"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEM"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '11:30:00':
                            print("true")
                            subject="DBMS"
                            return attendance_take_CE()
                        if timeStamp >= '11:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-3"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEM"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DS"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                if division_a=="B":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '11:30:00':
                            print("true")
                            subject="DBMS"
                            return attendance_take_CE()
                        if timeStamp >= '11:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-3"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEM"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DS"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-3_Tutorial"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEM"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DBMS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="DS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Maths-3"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DE_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-3"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DS"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="DBMS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
            if semester_a == "4":
                if division_a == "A":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OOPC++"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Maths-4"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OOPC++_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CN"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CO"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Maths-4_Tutorial"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CN"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CO_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "Maths-4"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OS"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CN_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CO"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OOPC++"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                if division_a == "B":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CN_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CO"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OOPC++"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CO_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "Maths-4"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OS"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OOPC++_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CN"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CO"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Maths-4_Tutorial"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CN"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OOPC++"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Maths-4"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
            if semester_a == "5":
                if division_a=="A":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="ADA_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="MPI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="OOPJ"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="OOPJ_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="MPI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="SP"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="MPI_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="ADA"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="SP"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="SP_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="OOPJ"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ADA"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '11:30:00':
                            print("true")
                            subject="ADA"
                            return attendance_take_CE()
                        if timeStamp >= '11:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="SP"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="OOPJ"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="MPI"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                if division_a=="B":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '11:30:00':
                            print("true")
                            subject = "ADA"
                            return attendance_take_CE()
                        if timeStamp >= '11:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "SP"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OOPJ"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "MPI"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "SP_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OOPJ"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "ADA"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OOPJ_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "MPI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "SP"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "MPI_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "ADA"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "SP"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "ADA_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "MPI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OOPJ"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
            if semester_a == "6":
                if division_a == "A":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "SE_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "A.Java"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "TOC"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "TOC_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "WT"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = ".Net"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "A.Java_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "SE"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = ".Net"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "WT_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "A.Java"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "SE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = ".Net_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "WT"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "A.Java"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                if division_a == "B":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = ".Net_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "WT"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "A.Java"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "WT_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "A.Java"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "SE"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "TOC_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "WT"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = ".Net"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "A.Java_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "SE"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = ".Net"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "SE_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "A.Java"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "TOC"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
            if semester_a == "7":
                if division_a == "A":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CD_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DMBI"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "INS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "MCWC"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CD"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "DMBI_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "MCWC"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "MCWC_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CD"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Project-1"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "DMBI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CD"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                if division_a == "B":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Project-1"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "DMBI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CD"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "MCWC_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CD"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "INS_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "MCWC"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CD"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "DMBI_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "MCWC"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CD_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DMBI"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
            if semester_a == "8":
                if division_a == "A":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "AI_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "Python"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Project-2"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Python_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "AI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Project-2"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Project-2"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "AI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Python"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Python_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '15:00:00':
                            subject = "AI_Lab"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        return render_template("holiday.html")
                if division_a == "B":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Python_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '15:00:00':
                            subject = "AI_Lab"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Project-2"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "AI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Python"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Python_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "AI"
                            return attendance_take_CE()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Project-2"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "AI_Lab"
                            return attendance_take_CE()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "Python"
                            return attendance_take_CE()
                        if timeStamp >= '17:00:00' and timeStamp <= '20:00:00':
                            subject = "Project-2"
                            return attendance_take_CE()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        return render_template("holiday.html")
        if department_a=="Information Technology":
            if semester_a=="1":
                if division_a=="A":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-1"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ES"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-1_Tutorial"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEE"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="EME"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EEE_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EME"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ES"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EME_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-1"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CPU"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CPU_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEE"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                if division_a=="B":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CPU_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEE"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EME_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-1"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CPU"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-1_Tutorial"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEE"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="EME"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EEE_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EME"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ES"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-1"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ES"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
            if semester_a=="2":
                if division_a=="A":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CPD_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EG"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="BE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EG_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-2"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CPD"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-2_Tutorial"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Physics"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="BE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="BE_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Physics"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="EG"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Physics_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EG"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Maths-2"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                if division_a=="B":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Physics_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EG"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Maths-2"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="BE_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Physics"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="EG"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="EG"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-2"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CPD"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CPD_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EG"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="BE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-2_Tutorial"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Physics"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="BE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
            if semester_a=="3":
                if division_a=="A":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="DBMS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DBMS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="DS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Maths-3"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DE_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-3"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-3_Tutorial"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEM"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '11:30:00':
                            print("true")
                            subject="DBMS"
                            return attendance_take_IT()
                        if timeStamp >= '11:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-3"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEM"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                if division_a=="B":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '11:30:00':
                            print("true")
                            subject="DBMS"
                            return attendance_take_IT()
                        if timeStamp >= '11:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-3"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEM"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="Maths-3_Tutorial"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="EEM"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DBMS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="DS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="Maths-3"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DE_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="Maths-3"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="DS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="DBMS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="DE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
            if semester_a == "4":
                if division_a == "A":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OOPC++"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Maths-4"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OOPC++_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CN"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CO"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Maths-4_Tutorial"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CN"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CO_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "Maths-4"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CN_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CO"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OOPC++"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                if division_a == "B":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CN_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CO"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OOPC++"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CO_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "Maths-4"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "OS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OOPC++_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CN"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CO"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Maths-4_Tutorial"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CN"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OOPC++"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Maths-4"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
            if semester_a=="5":
                if division_a=="A":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="ADA_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="CG"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="OOPJ"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="OOPJ_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="CG"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="SP"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="CG_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="ADA"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="SP"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="SP_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="OOPJ"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="ADA"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '11:30:00':
                            print("true")
                            subject="ADA"
                            return attendance_take_IT()
                        if timeStamp >= '11:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject="SP"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject="OOPJ"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject="CG"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                if division_a=="B":
                    if day_names=="Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '11:30:00':
                            print("true")
                            subject = "ADA"
                            return attendance_take_IT()
                        if timeStamp >= '11:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "SP"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OOPJ"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "CG"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "SP_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "OOPJ"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "ADA"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "OOPJ_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CG"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "SP"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "CG_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "ADA"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "SP"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names=="Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "ADA_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "CG"
                            return attendance_take_IT()
                        if timeStamp >= '20:00:00' and timeStamp <= '22:00:00':
                            subject = "OOPJ"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
            if semester_a == "6":
                if division_a == "A":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "SE_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "A.Java"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DCDR"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "DCDR_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "WT"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = ".Net"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "A.Java_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "SE"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = ".Net"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "WT_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "A.Java"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "SE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = ".Net_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "WT"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "A.Java"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                if division_a == "B":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = ".Net_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "WT"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "A.Java"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "WT_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "A.Java"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "SE"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "DCDR_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "WT"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = ".Net"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "A.Java_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "SE"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = ".Net"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "SE_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "A.Java"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DCDR"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
            if semester_a == "7":
                if division_a == "A":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "DDBMS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DMBI"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "INS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "MCWC"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DDBMS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "DMBI_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "MCWC"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "MCWC_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DDBMS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Project-1"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "DMBI"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DDBMS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                if division_a == "B":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Project-1"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "DMBI"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DDBMS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "MCWC_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DDBMS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "INS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "MCWC"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DDBMS"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "DMBI_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "MCWC"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "DDBMS_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "INS"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "DMBI"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
            if semester_a == "8":
                if division_a == "A":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "AI_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "Python"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Project-2"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Python_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "AI"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Project-2"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Project-2"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "AI"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Python"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Python_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '15:00:00':
                            subject = "AI_Lab"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        return render_template("holiday.html")
                if division_a == "B":
                    if day_names == "Monday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Python_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '15:00:00':
                            subject = "AI_Lab"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Tuesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Project-2"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "AI"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Python"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Wednesday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "Python_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "AI"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Project-2"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Thursday":
                        if timeStamp >= '10:30:00' and timeStamp <= '12:30:00':
                            print("true")
                            subject = "AI_Lab"
                            return attendance_take_IT()
                        if timeStamp >= '13:00:00' and timeStamp <= '14:00:00':
                            subject = "Python"
                            return attendance_take_IT()
                        if timeStamp >= '14:00:00' and timeStamp <= '15:00:00':
                            subject = "Project-2"
                            return attendance_take_IT()
                        else:
                            return render_template("Closed.html")
                    if day_names == "Friday":
                        return render_template("holiday.html")



def attendance_take_CE():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
        # parent_dir1 = "TrainingImageLabel"
        # directory1 = str(department_a) + "\Semester_" + str(semester_a) + "\ " + str(division_a)
        # path_yml = os.path.join(parent_dir1, directory1)
        exists3 = os.path.isfile("TrainingImageLabel/"+str(department_a) + "/Semester_" + str(semester_a) + "/DIV_" + str(division_a)+"/Trainner.yml")
        if exists3:
            print("True--123")
            recognizer.read("TrainingImageLabel/"+str(department_a) + "/Semester_" + str(semester_a) + "/DIV_" + str(division_a)+"/Trainner.yml")
        else:
            print("elese")
            return render_template('face_detect.html')
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath);
        df = pd.read_csv("StudentDetails\Computer\St_Details_CE_" + semester_a + "_" + division_a + ".csv")
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Enrollment', 'name', 'Date']
        exists1 = os.path.isfile("StudentDetails\Computer\St_Details_CE_" + semester_a + "_" + division_a + ".csv")
        if exists1:
            attendance = pd.DataFrame(columns=col_names)
            attendance1 = pd.DataFrame(columns=col_names)
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if (conf < 50):
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Enrollment'] == Id]['name'].values
                    bb = str(aa)
                    # bb=bb[2:-2]
                    # tt = str(Id) + "_" + str(aa)
                    attendance1.loc[len(attendance1)] = [str(Id), bb, str(date)]
                    attendance = [str(Id), bb, str(date)]
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(im, (x, y + h - 35), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(im, bb, (x + 6, y + h - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    Id = 'Unknown'
                    bb = str(Id)
                    # if (conf > 10):
                    #     noOfFile = len(os.listdir("ImagesUnknown")) + 1
                    #     cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.rectangle(im, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)
                    cv2.putText(im, 'Unknown', (x + 6, y + h - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
            attendance1 = attendance1.drop_duplicates(subset=['Enrollment'], keep='first')
            cv2.imshow('im', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        # timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        # Hour, Minute, Second = timeStamp.split(":")
        # fileName = "Attendance\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
        # attendance.to_csv(fileName, index=False)
        # cam.release()
        # cv2.destroyAllWindows()
        # return render_template('face_detect.html')
        exists = os.path.isfile("Cam_data\Attendance_" + date + "_" + subject + ".csv")
        if exists:
            with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", newline='') as csvFile1:
                atd = pd.read_csv(csvFile1)
                atd_l = list(atd['name'])
                # Id_l=set(Id)
            if str(aa) not in atd_l:
                with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", 'a+') as csvFile1:
                    writer = csv.writer(csvFile1)
                    writer.writerow(attendance)
                    # attendance.to_csv(csvFile1, index=False)
                csvFile1.close()
        else:
            with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", 'a+') as csvFile1:
                attendance1.to_csv(csvFile1, index=False)
            csvFile1.close()
    except FileNotFoundError:
        return render_template("FileNotFound.html")
    except UnboundLocalError:
        return render_template("not_registered.html")
    cam.release()
    cv2.destroyAllWindows()

    exists_dir = os.path.isdir(
        "Attendance_csv\ " + str(department_a) + "\Semester_" + str(semester_a) + "\DIV_ " + str(division_a))
    if not exists_dir:
        try:
            parent_dir = "Attendance_csv"
            directory = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path1 = os.path.join(parent_dir, directory)
            os.makedirs(path1, exist_ok=True)
            with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", newline='') as csvFile1:
                Af = pd.read_csv(csvFile1)
                Af = Af.drop_duplicates(subset=['Enrollment'], keep='first')
            with open("StudentDetails\Computer\St_Details_CE_" + semester_a + "_" + division_a + ".csv",
                      newline='') as csvFile:
                st = pd.read_csv(csvFile)
            st_l_a = set(list(st['Enrollment']))
            Af_l = set(list(Af['Enrollment']))
            if Af_l.issubset(st_l_a):

                list_Pr_a = list(Af['Enrollment'][Af.Enrollment.isin(st.Enrollment)])
                list_Pr_b = list(Af['name'][Af.Enrollment.isin(st.Enrollment)])
                len_pr = len(list_Pr_a)
                j = 0
                while j <= len_pr - 1:
                    Status = "Present"
                    # Enrollment= Af['Enrollment'].to_string(index=False)
                    # Enrollment_A = Af['Enrollment']
                    Enrollment = list_Pr_a[j]
                    # name_A = Af['name'].to_string(index=False)
                    name_A = list_Pr_b[j]
                    col_names_value = [Enrollment, name_A, Status]
                    col_names = ['Enrollment', 'name', date]
                    # col_names1=col_names.append(timeStamp)
                    # col_names_value1=col_names_value.append(Status)
                    print(col_names_value)
                    existsA = os.path.isfile(os.path.join(path1, subject + ".csv"))
                    if existsA:
                        with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                            df = pd.read_csv(atFinal)
                        st_l = set(list(st['Enrollment']))
                        df_l = set(list(df['Enrollment']))
                        if st_l == df_l:
                            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                                # writer=csv.writer(atFinal)
                                # writer.writerow(col_names_value)
                                df = pd.read_csv(atFinal)
                                # Enrollment_A=Af['Enrollment'][df.Enrollment.isin(Af.Enrollment)].values
                                # j=0
                                df[date] = Status
                                # while j < len(df['Enrollment']):
                                #     if df['Enrollment'][j] == Enrollment_A:
                                #         # print("True")
                                #         df[timeStamp].loc[df['Enrollment'][j]==Enrollment_A]=Status
                                #     j+=1
                                # df.head()
                                df1 = pd.DataFrame(df)
                                df1.to_csv(os.path.join(path1, subject + ".csv"), index=False)
                            atFinal.close()
                        elif st_l != df_l:
                            with open(os.path.join(path1, subject + ".csv"), 'a+') as atFinal:
                                writer = csv.writer(atFinal)
                                writer.writerow(col_names_value)
                            atFinal.close()
                    else:
                        with open(os.path.join(path1, subject + ".csv"), "a+") as atFinal:
                            writer = csv.writer(atFinal)
                            writer.writerow(col_names)
                            writer.writerow(col_names_value)

                        atFinal.close()
                    j += 1
            # print(df)
            if st['Enrollment'].to_string(index=False) not in Af['Enrollment'].to_string(index=False):
                # print("true")
                len_a = len(st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)])
                list_a = list(st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)])
                list_b = list(st['name'][~st.Enrollment.isin(Af.Enrollment)])
                # print(len_a)
                i = 0
                while i <= len_a - 1:
                    Status = "absent"
                    # Enrollment = st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)].values
                    # name = st['name'][~st.Enrollment.isin(Af.Enrollment)].values
                    Enrollment_B = list_a[i]
                    name = list_b[i]
                    # print(list_a[i])
                    # print(list_b[i])

                    # name=name.to_string(index=False)
                    # Enrollment=Enrollment.to_string(index=False)
                    col_names_value = [Enrollment_B, name, Status]
                    print(col_names_value)
                    col_names = ['Enrollment', 'name', date]
                    existsA = os.path.isfile(os.path.join(path1, subject + ".csv"))
                    if existsA:
                        with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                            # writer=csv.writer(atFinal)
                            # writer.writerow(col_names_value)
                            df = pd.read_csv(atFinal)
                        st_l = set(list(st['Enrollment']))
                        df_l = set(list(df['Enrollment']))
                        if st_l != df_l:
                            with open(os.path.join(path1, subject + ".csv"), 'a+') as atFinal:
                                writer = csv.writer(atFinal)
                                writer.writerow(col_names_value)
                                # df2=pd.read_csv(atFinal)
                                # ab=pd.read_csv(atFinal)
                                # df2=pd.DataFrame(ab)
                                # df1[timeStamp] = Status
                                # Af.head()
                            atFinal.close()
                        elif st_l == df_l:
                            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                                cf = pd.read_csv(atFinal)

                                # cf[timeStamp] = ""
                                cf[date].loc[cf['Enrollment'] == Enrollment_B] = Status
                                # df.head()
                                cf1 = pd.DataFrame(cf)
                                cf1.to_csv(os.path.join(path1, subject + ".csv"), index=False)
                    else:
                        with open(os.path.join(path1, subject + ".csv"), "a+") as atFinal:
                            writer = csv.writer(atFinal)
                            writer.writerow(col_names)
                            writer.writerow(col_names_value)
                        atFinal.close()
                    i += 1
        except FileNotFoundError:
            return render_template("FileNotFound.html")
        exists_exe = os.path.isdir("Attendance_excel\ " + str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a))
        if not exists_exe:
            parent_excel = "Attendance_excel"
            dir_excel = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path_execl = os.path.join(parent_excel, dir_excel)
            os.makedirs(path_execl, exist_ok=True)
            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                ef = pd.read_csv(atFinal)
                Excel_File = pd.ExcelWriter(os.path.join(path_execl, subject + ".xlsx"))
                ef.to_excel(Excel_File, index=False)
                Excel_File.save()
        elif exists_exe:
            parent_excel = "Attendance_excel"
            dir_excel = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path_execl = os.path.join(parent_excel, dir_excel)
            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                ef = pd.read_csv(atFinal)
                Excel_File = pd.ExcelWriter(os.path.join(path_execl, subject + ".xlsx"))
                ef.to_excel(Excel_File, index=False)
                Excel_File.save()
    elif exists_dir:
        try:
            parent_dir = "Attendance_csv"
            directory = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path1 = os.path.join(parent_dir, directory)
            with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", newline='') as csvFile1:
                Af = pd.read_csv(csvFile1)
                Af = Af.drop_duplicates(subset=['Enrollment'], keep='first')
            with open("StudentDetails\Computer\St_Details_CE_" + semester_a + "_" + division_a + ".csv",
                      newline='') as csvFile:
                st = pd.read_csv(csvFile)
            st_l_a = set(list(st['Enrollment']))
            Af_l = set(list(Af['Enrollment']))
            if Af_l.issubset(st_l_a):

                list_Pr_a = list(Af['Enrollment'][Af.Enrollment.isin(st.Enrollment)])
                list_Pr_b = list(Af['name'][Af.Enrollment.isin(st.Enrollment)])
                len_pr = len(list_Pr_a)
                j = 0
                while j <= len_pr - 1:
                    Status = "Present"
                    # Enrollment= Af['Enrollment'].to_string(index=False)
                    # Enrollment_A = Af['Enrollment']
                    Enrollment = list_Pr_a[j]
                    # name_A = Af['name'].to_string(index=False)
                    name_A = list_Pr_b[j]
                    col_names_value = [Enrollment, name_A, Status]
                    col_names = ['Enrollment', 'name', date]
                    # col_names1=col_names.append(timeStamp)
                    # col_names_value1=col_names_value.append(Status)
                    print(col_names_value)
                    existsA = os.path.isfile(os.path.join(path1, subject + ".csv"))
                    if existsA:
                        with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                            df = pd.read_csv(atFinal)
                        st_l = set(list(st['Enrollment']))
                        df_l = set(list(df['Enrollment']))
                        if st_l == df_l:
                            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                                # writer=csv.writer(atFinal)
                                # writer.writerow(col_names_value)
                                df = pd.read_csv(atFinal)
                                # Enrollment_A=Af['Enrollment'][df.Enrollment.isin(Af.Enrollment)].values
                                # j=0
                                df[date] = Status
                                # while j < len(df['Enrollment']):
                                #     if df['Enrollment'][j] == Enrollment_A:
                                #         # print("True")
                                #         df[timeStamp].loc[df['Enrollment'][j]==Enrollment_A]=Status
                                #     j+=1
                                # df.head()
                                df1 = pd.DataFrame(df)
                                df1.to_csv(os.path.join(path1, subject + ".csv"), index=False)
                            atFinal.close()
                        elif st_l != df_l:
                            with open(os.path.join(path1, subject + ".csv"), 'a+') as atFinal:
                                writer = csv.writer(atFinal)
                                writer.writerow(col_names_value)
                            atFinal.close()
                    else:
                        with open(os.path.join(path1, subject + ".csv"), "a+") as atFinal:
                            writer = csv.writer(atFinal)
                            writer.writerow(col_names)
                            writer.writerow(col_names_value)

                        atFinal.close()
                    j += 1
            # print(df)
            if st['Enrollment'].to_string(index=False) not in Af['Enrollment'].to_string(index=False):
                # print("true")
                len_a = len(st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)])
                list_a = list(st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)])
                list_b = list(st['name'][~st.Enrollment.isin(Af.Enrollment)])
                # print(len_a)
                i = 0
                while i <= len_a - 1:
                    Status = "absent"
                    # Enrollment = st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)].values
                    # name = st['name'][~st.Enrollment.isin(Af.Enrollment)].values
                    Enrollment_B = list_a[i]
                    name = list_b[i]
                    # print(list_a[i])
                    # print(list_b[i])

                    # name=name.to_string(index=False)
                    # Enrollment=Enrollment.to_string(index=False)
                    col_names_value = [Enrollment_B, name, Status]
                    print(col_names_value)
                    col_names = ['Enrollment', 'name', date]
                    existsA = os.path.isfile(os.path.join(path1, subject + ".csv"))
                    if existsA:
                        with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                            # writer=csv.writer(atFinal)
                            # writer.writerow(col_names_value)
                            df = pd.read_csv(atFinal)
                        st_l = set(list(st['Enrollment']))
                        df_l = set(list(df['Enrollment']))
                        if st_l != df_l:
                            with open(os.path.join(path1, subject + ".csv"), 'a+') as atFinal:
                                writer = csv.writer(atFinal)
                                writer.writerow(col_names_value)
                                # df2=pd.read_csv(atFinal)
                                # ab=pd.read_csv(atFinal)
                                # df2=pd.DataFrame(ab)
                                # df1[timeStamp] = Status
                                # Af.head()
                            atFinal.close()
                        elif st_l == df_l:
                            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                                cf = pd.read_csv(atFinal)

                                # cf[timeStamp] = ""
                                cf[date].loc[cf['Enrollment'] == Enrollment_B] = Status
                                # df.head()
                                cf1 = pd.DataFrame(cf)
                                cf1.to_csv(os.path.join(path1, subject + ".csv"), index=False)
                    else:
                        with open(os.path.join(path1, subject + ".csv"), "a+") as atFinal:
                            writer = csv.writer(atFinal)
                            writer.writerow(col_names)
                            writer.writerow(col_names_value)
                        atFinal.close()
                    i += 1
        except FileNotFoundError:
            return render_template("FileNotFound.html")
        exists_exe = os.path.isdir("Attendance_excel\ " + str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a))
        if not exists_exe:
            parent_excel = "Attendance_excel"
            dir_excel = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path_execl = os.path.join(parent_excel, dir_excel)
            os.makedirs(path_execl, exist_ok=True)
            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                ef = pd.read_csv(atFinal)
                Excel_File = pd.ExcelWriter(os.path.join(path_execl, subject + ".xlsx"))
                ef.to_excel(Excel_File, index=False)
                Excel_File.save()
        elif exists_exe:
            parent_excel = "Attendance_excel"
            dir_excel = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path_execl = os.path.join(parent_excel, dir_excel)
            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                ef = pd.read_csv(atFinal)
                Excel_File = pd.ExcelWriter(os.path.join(path_execl, subject + ".xlsx"))
                ef.to_excel(Excel_File, index=False)
                Excel_File.save()

    return render_template('face_detect.html')


def attendance_take_IT():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
        # parent_dir1 = "TrainingImageLabel"
        # directory1 = str(department_a) + "\Semester_" + str(semester_a) + "\ " + str(division_a)
        # path_yml = os.path.join(parent_dir1, directory1)
        exists3 = os.path.isfile("TrainingImageLabel/"+str(department_a) + "/Semester_" + str(semester_a) + "/DIV_" + str(division_a)+"/Trainner.yml")
        if exists3:
            print("True--123")
            recognizer.read("TrainingImageLabel/"+str(department_a) + "/Semester_" + str(semester_a) + "/DIV_" + str(division_a)+"/Trainner.yml")
        else:
            print("elese")
            return render_template('face_detect.html')
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath);
        df = pd.read_csv("StudentDetails\IT\St_Details_IT_" + semester_a + "_" + division_a + ".csv")
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Enrollment', 'name', 'Date']
        exists1 = os.path.isfile("StudentDetails\IT\St_Details_IT_" + semester_a + "_" + division_a + ".csv")
        if exists1:
            attendance = pd.DataFrame(columns=col_names)
            attendance1 = pd.DataFrame(columns=col_names)
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if (conf < 50):
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    aa = df.loc[df['Enrollment'] == Id]['name'].values
                    bb = str(aa)
                    # bb=bb[2:-2]
                    # tt = str(Id) + "_" + str(aa)
                    attendance1.loc[len(attendance1)] = [str(Id), bb, str(date)]
                    attendance = [str(Id), bb, str(date)]
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(im, (x, y + h - 35), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(im, bb, (x + 6, y + h - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    Id = 'Unknown'
                    bb = str(Id)
                    # if (conf > 10):
                    #     noOfFile = len(os.listdir("ImagesUnknown")) + 1
                    #     cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.rectangle(im, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)
                    cv2.putText(im, 'Unknown', (x + 6, y + h - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
            attendance1 = attendance1.drop_duplicates(subset=['Enrollment'], keep='first')
            cv2.imshow('im', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        # timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        # Hour, Minute, Second = timeStamp.split(":")
        # fileName = "Attendance\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
        # attendance.to_csv(fileName, index=False)
        # cam.release()
        # cv2.destroyAllWindows()
        # return render_template('face_detect.html')
        exists = os.path.isfile("Cam_data\Attendance_" + date + "_" + subject + ".csv")
        if exists:
            with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", newline='') as csvFile1:
                atd = pd.read_csv(csvFile1)
                atd_l = list(atd['name'])
                # Id_l=set(Id)
            if str(aa) not in atd_l:
                with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", 'a+') as csvFile1:
                    writer = csv.writer(csvFile1)
                    writer.writerow(attendance)
                    # attendance.to_csv(csvFile1, index=False)
                csvFile1.close()
        else:
            with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", 'a+') as csvFile1:
                attendance1.to_csv(csvFile1, index=False)
            csvFile1.close()
    except FileNotFoundError:
        return render_template("FileNotFound.html")
    except UnboundLocalError:
        return render_template("not_registered.html")
    cam.release()
    cv2.destroyAllWindows()

    exists_dir = os.path.isdir("Attendance_csv\ " + str(department_a) + "\Semester_" + str(semester_a) + "\DIV_ " + str(division_a))
    if not exists_dir:
        try:
            parent_dir = "Attendance_csv"
            directory = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path1 = os.path.join(parent_dir, directory)
            os.makedirs(path1, exist_ok=True)
            with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", newline='') as csvFile1:
                Af = pd.read_csv(csvFile1)
                Af = Af.drop_duplicates(subset=['Enrollment'], keep='first')
            with open("StudentDetails\IT\St_Details_IT_" + semester_a + "_" + division_a + ".csv",
                      newline='') as csvFile:
                st = pd.read_csv(csvFile)
            st_l_a = set(list(st['Enrollment']))
            Af_l = set(list(Af['Enrollment']))
            if Af_l.issubset(st_l_a):

                list_Pr_a = list(Af['Enrollment'][Af.Enrollment.isin(st.Enrollment)])
                list_Pr_b = list(Af['name'][Af.Enrollment.isin(st.Enrollment)])
                len_pr = len(list_Pr_a)
                j = 0
                while j <= len_pr - 1:
                    Status = "Present"
                    # Enrollment= Af['Enrollment'].to_string(index=False)
                    # Enrollment_A = Af['Enrollment']
                    Enrollment = list_Pr_a[j]
                    # name_A = Af['name'].to_string(index=False)
                    name_A = list_Pr_b[j]
                    col_names_value = [Enrollment, name_A, Status]
                    col_names = ['Enrollment', 'name', date]
                    # col_names1=col_names.append(timeStamp)
                    # col_names_value1=col_names_value.append(Status)
                    print(col_names_value)
                    existsA = os.path.isfile(os.path.join(path1, subject + ".csv"))
                    if existsA:
                        with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                            df = pd.read_csv(atFinal)
                        st_l = set(list(st['Enrollment']))
                        df_l = set(list(df['Enrollment']))
                        if st_l == df_l:
                            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                                # writer=csv.writer(atFinal)
                                # writer.writerow(col_names_value)
                                df = pd.read_csv(atFinal)
                                # Enrollment_A=Af['Enrollment'][df.Enrollment.isin(Af.Enrollment)].values
                                # j=0
                                df[date] = Status
                                # while j < len(df['Enrollment']):
                                #     if df['Enrollment'][j] == Enrollment_A:
                                #         # print("True")
                                #         df[timeStamp].loc[df['Enrollment'][j]==Enrollment_A]=Status
                                #     j+=1
                                # df.head()
                                df1 = pd.DataFrame(df)
                                df1.to_csv(os.path.join(path1, subject + ".csv"), index=False)
                            atFinal.close()
                        elif st_l != df_l:
                            with open(os.path.join(path1, subject + ".csv"), 'a+') as atFinal:
                                writer = csv.writer(atFinal)
                                writer.writerow(col_names_value)
                            atFinal.close()
                    else:
                        with open(os.path.join(path1, subject + ".csv"), "a+") as atFinal:
                            writer = csv.writer(atFinal)
                            writer.writerow(col_names)
                            writer.writerow(col_names_value)

                        atFinal.close()
                    j += 1
            # print(df)
            if st['Enrollment'].to_string(index=False) not in Af['Enrollment'].to_string(index=False):
                # print("true")
                len_a = len(st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)])
                list_a = list(st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)])
                list_b = list(st['name'][~st.Enrollment.isin(Af.Enrollment)])
                # print(len_a)
                i = 0
                while i <= len_a - 1:
                    Status = "absent"
                    # Enrollment = st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)].values
                    # name = st['name'][~st.Enrollment.isin(Af.Enrollment)].values
                    Enrollment_B = list_a[i]
                    name = list_b[i]
                    # print(list_a[i])
                    # print(list_b[i])

                    # name=name.to_string(index=False)
                    # Enrollment=Enrollment.to_string(index=False)
                    col_names_value = [Enrollment_B, name, Status]
                    print(col_names_value)
                    col_names = ['Enrollment', 'name', date]
                    existsA = os.path.isfile(os.path.join(path1, subject + ".csv"))
                    if existsA:
                        with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                            # writer=csv.writer(atFinal)
                            # writer.writerow(col_names_value)
                            df = pd.read_csv(atFinal)
                        st_l = set(list(st['Enrollment']))
                        df_l = set(list(df['Enrollment']))
                        if st_l != df_l:
                            with open(os.path.join(path1, subject + ".csv"), 'a+') as atFinal:
                                writer = csv.writer(atFinal)
                                writer.writerow(col_names_value)
                                # df2=pd.read_csv(atFinal)
                                # ab=pd.read_csv(atFinal)
                                # df2=pd.DataFrame(ab)
                                # df1[timeStamp] = Status
                                # Af.head()
                            atFinal.close()
                        elif st_l == df_l:
                            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                                cf = pd.read_csv(atFinal)

                                # cf[timeStamp] = ""
                                cf[date].loc[cf['Enrollment'] == Enrollment_B] = Status
                                # df.head()
                                cf1 = pd.DataFrame(cf)
                                cf1.to_csv(os.path.join(path1, subject + ".csv"), index=False)
                    else:
                        with open(os.path.join(path1, subject + ".csv"), "a+") as atFinal:
                            writer = csv.writer(atFinal)
                            writer.writerow(col_names)
                            writer.writerow(col_names_value)
                        atFinal.close()
                    i += 1
        except FileNotFoundError:
            return render_template("FileNotFound.html")
        exists_exe = os.path.isdir("Attendance_excel\ " + str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a))
        if not exists_exe:
            parent_excel = "Attendance_excel"
            dir_excel = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path_execl = os.path.join(parent_excel, dir_excel)
            os.makedirs(path_execl, exist_ok=True)
            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                ef = pd.read_csv(atFinal)
                Excel_File = pd.ExcelWriter(os.path.join(path_execl, subject + ".xlsx"))
                ef.to_excel(Excel_File, index=False)
                Excel_File.save()
        elif exists_exe:
            parent_excel = "Attendance_excel"
            dir_excel = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path_execl = os.path.join(parent_excel, dir_excel)
            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                ef = pd.read_csv(atFinal)
                Excel_File = pd.ExcelWriter(os.path.join(path_execl, subject + ".xlsx"))
                ef.to_excel(Excel_File, index=False)
                Excel_File.save()
    elif exists_dir:
        try:
            parent_dir = "Attendance_csv"
            directory = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path1 = os.path.join(parent_dir, directory)
            with open("Cam_data\Attendance_" + date + "_" + subject + ".csv", newline='') as csvFile1:
                Af = pd.read_csv(csvFile1)
                Af = Af.drop_duplicates(subset=['Enrollment'], keep='first')
            with open("StudentDetails\IT\St_Details_IT_" + semester_a + "_" + division_a + ".csv",
                      newline='') as csvFile:
                st = pd.read_csv(csvFile)
            st_l_a = set(list(st['Enrollment']))
            Af_l = set(list(Af['Enrollment']))
            if Af_l.issubset(st_l_a):

                list_Pr_a = list(Af['Enrollment'][Af.Enrollment.isin(st.Enrollment)])
                list_Pr_b = list(Af['name'][Af.Enrollment.isin(st.Enrollment)])
                len_pr = len(list_Pr_a)
                j = 0
                while j <= len_pr - 1:
                    Status = "Present"
                    # Enrollment= Af['Enrollment'].to_string(index=False)
                    # Enrollment_A = Af['Enrollment']
                    Enrollment = list_Pr_a[j]
                    # name_A = Af['name'].to_string(index=False)
                    name_A = list_Pr_b[j]
                    col_names_value = [Enrollment, name_A, Status]
                    col_names = ['Enrollment', 'name', date]
                    # col_names1=col_names.append(timeStamp)
                    # col_names_value1=col_names_value.append(Status)
                    print(col_names_value)
                    existsA = os.path.isfile(os.path.join(path1, subject + ".csv"))
                    if existsA:
                        with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                            df = pd.read_csv(atFinal)
                        st_l = set(list(st['Enrollment']))
                        df_l = set(list(df['Enrollment']))
                        if st_l == df_l:
                            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                                # writer=csv.writer(atFinal)
                                # writer.writerow(col_names_value)
                                df = pd.read_csv(atFinal)
                                # Enrollment_A=Af['Enrollment'][df.Enrollment.isin(Af.Enrollment)].values
                                # j=0
                                df[date] = Status
                                # while j < len(df['Enrollment']):
                                #     if df['Enrollment'][j] == Enrollment_A:
                                #         # print("True")
                                #         df[timeStamp].loc[df['Enrollment'][j]==Enrollment_A]=Status
                                #     j+=1
                                # df.head()
                                df1 = pd.DataFrame(df)
                                df1.to_csv(os.path.join(path1, subject + ".csv"), index=False)
                            atFinal.close()
                        elif st_l != df_l:
                            with open(os.path.join(path1, subject + ".csv"), 'a+') as atFinal:
                                writer = csv.writer(atFinal)
                                writer.writerow(col_names_value)
                            atFinal.close()
                    else:
                        with open(os.path.join(path1, subject + ".csv"), "a+") as atFinal:
                            writer = csv.writer(atFinal)
                            writer.writerow(col_names)
                            writer.writerow(col_names_value)

                        atFinal.close()
                    j += 1
            # print(df)
            if st['Enrollment'].to_string(index=False) not in Af['Enrollment'].to_string(index=False):
                # print("true")
                len_a = len(st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)])
                list_a = list(st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)])
                list_b = list(st['name'][~st.Enrollment.isin(Af.Enrollment)])
                # print(len_a)
                i = 0
                while i <= len_a - 1:
                    Status = "absent"
                    # Enrollment = st['Enrollment'][~st.Enrollment.isin(Af.Enrollment)].values
                    # name = st['name'][~st.Enrollment.isin(Af.Enrollment)].values
                    Enrollment_B = list_a[i]
                    name = list_b[i]
                    # print(list_a[i])
                    # print(list_b[i])

                    # name=name.to_string(index=False)
                    # Enrollment=Enrollment.to_string(index=False)
                    col_names_value = [Enrollment_B, name, Status]
                    print(col_names_value)
                    col_names = ['Enrollment', 'name', date]
                    existsA = os.path.isfile(os.path.join(path1, subject + ".csv"))
                    if existsA:
                        with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                            # writer=csv.writer(atFinal)
                            # writer.writerow(col_names_value)
                            df = pd.read_csv(atFinal)
                        st_l = set(list(st['Enrollment']))
                        df_l = set(list(df['Enrollment']))
                        if st_l != df_l:
                            with open(os.path.join(path1, subject + ".csv"), 'a+') as atFinal:
                                writer = csv.writer(atFinal)
                                writer.writerow(col_names_value)
                                # df2=pd.read_csv(atFinal)
                                # ab=pd.read_csv(atFinal)
                                # df2=pd.DataFrame(ab)
                                # df1[timeStamp] = Status
                                # Af.head()
                            atFinal.close()
                        elif st_l == df_l:
                            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                                cf = pd.read_csv(atFinal)

                                # cf[timeStamp] = ""
                                cf[date].loc[cf['Enrollment'] == Enrollment_B] = Status
                                # df.head()
                                cf1 = pd.DataFrame(cf)
                                cf1.to_csv(os.path.join(path1, subject + ".csv"), index=False)
                    else:
                        with open(os.path.join(path1, subject + ".csv"), "a+") as atFinal:
                            writer = csv.writer(atFinal)
                            writer.writerow(col_names)
                            writer.writerow(col_names_value)
                        atFinal.close()
                    i += 1
        except FileNotFoundError:
            return render_template("FileNotFound.html")
        exists_exe = os.path.isdir("Attendance_excel\ " + str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a))
        if not exists_exe:
            parent_excel = "Attendance_excel"
            dir_excel = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path_execl = os.path.join(parent_excel, dir_excel)
            os.makedirs(path_execl, exist_ok=True)
            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                ef = pd.read_csv(atFinal)
                Excel_File = pd.ExcelWriter(os.path.join(path_execl, subject + ".xlsx"))
                ef.to_excel(Excel_File, index=False)
                Excel_File.save()
        elif exists_exe:
            parent_excel = "Attendance_excel"
            dir_excel = str(department_a) + "\Semester_" + str(semester_a) + "\DIV_" + str(division_a)
            path_execl = os.path.join(parent_excel, dir_excel)
            with open(os.path.join(path1, subject + ".csv"), newline='') as atFinal:
                ef = pd.read_csv(atFinal)
                Excel_File = pd.ExcelWriter(os.path.join(path_execl, subject + ".xlsx"))
                ef.to_excel(Excel_File, index=False)
                Excel_File.save()

    return render_template('face_detect.html')



@app.route("/St_details", methods=['POST'])
def St_details():
    global semester_b, department_b, division_b
    if request.method == "POST":
        result=[]
        semester_b = request.form["sem1"]
        department_b = request.form["dept1"]
        division_b = request.form["div1"]
        try:
            if department_b=="Computer":
                if semester_b=="1":
                   return st_table_CE()
                if semester_b=="2":
                   return st_table_CE()
                if semester_b=="3":
                   return st_table_CE()
                if semester_b=="4":
                   return st_table_CE()
                if semester_b=="5":
                   return st_table_CE()
                if semester_b=="6":
                   return st_table_CE()
                if semester_b=="7":
                   return st_table_CE()
                if semester_b=="8":
                   return st_table_CE()
            if department_b=="Information Technology":
                if semester_b=="1":
                   return st_table_IT()
                if semester_b=="2":
                   return st_table_IT()
                if semester_b=="3":
                   return st_table_IT()
                if semester_b=="4":
                   return st_table_IT()
                if semester_b=="5":
                   return st_table_IT()
                if semester_b=="6":
                   return st_table_IT()
                if semester_b=="7":
                   return st_table_IT()
                if semester_b=="8":
                   return st_table_IT()
        except FileNotFoundError:
            return render_template("FileNotFound.html")

def st_table_CE():
    if division_b == "A":
        result = []
        with open('StudentDetails\Computer\St_Details_CE_' + str(semester_b) + '_' + str(division_b) + '.csv',
                  'r') as csvFile_s:
            csv_reader = csv.reader(csvFile_s)
            for row in csv_reader:
                result.append(row)
            heading = result.pop(0)
            # i = 2
            # while i <= len(result) - 1:
            #     data = result[i]
            #     i += 1
    elif division_b=="B":
        result = []
        with open('StudentDetails\Computer\St_Details_CE_' + str(semester_b) + '_' + str(division_b) + '.csv',
                  'r') as csvFile_s:
            csv_reader = csv.reader(csvFile_s)
            for row in csv_reader:
                result.append(row)
            heading = result.pop(0)
            # i = 2
            # while i <= len(result) - 1:
            #     data = result[i]
            #     i += 1
    return render_template("St_details_table.html", heading=heading, result=result)

def st_table_IT():
    if division_b == "A":
        result = []
        with open('StudentDetails\IT\St_Details_IT_' + str(semester_b) + '_' + str(division_b) + '.csv',
                  'r') as csvFile_s:
            csv_reader = csv.reader(csvFile_s)
            for row in csv_reader:
                result.append(row)
            heading = result.pop(0)
            # i = 2
            # while i <= len(result) - 1:
            #     data = result[i]
            #     i += 1
    elif division_b=="B":
        result = []
        with open('StudentDetails\IT\St_Details_IT_' + str(semester_b) + '_' + str(division_b) + '.csv',
                  'r') as csvFile_s:
            csv_reader = csv.reader(csvFile_s)
            for row in csv_reader:
                result.append(row)
            heading = result.pop(0)
            # i = 2
            # while i <= len(result) - 1:
            #     data = result[i]
            #     i += 1
    return render_template("St_details_table.html", heading=heading, result=result)

@app.route("/get_st_data")
def index_1():
    return render_template("St_details_interface.html")

@app.route('/display_attendance')
def index_2():
    return render_template("display_attendance.html")

@app.route('/print_subject',methods=['POST'])
def print_subject():
    global semester_b, department_b, division_b,dir_list,file_path,final_dir_list
    final_dir_list=[]
    if request.method == "POST":
        try:
            semester_b = request.form["sem1"]
            department_b = request.form["dept1"]
            division_b = request.form["div1"]
            # file_path="Attendance_csv\ " + str(department_b) + "\Semester_" + str(semester_b) + "\DIV_ " + str(division_b)
            parent_dir = "Attendance_csv"
            directory = str(department_b) + "\Semester_" + str(semester_b) + "\DIV_" + str(division_b)
            file_path = os.path.join(parent_dir, directory)
            dir_list=os.listdir(file_path)
            for X in dir_list:
                y=X.split(".")[0]
                final_dir_list.append(y)
        except FileNotFoundError:
            return render_template("FileNotFound.html")
        except IsADirectoryError:
            return render_template("FileNotFound.html")
    return render_template("subject.html",final_dir_list=final_dir_list)

@app.route('/print_attendance_data',methods=['POST'])
def print_attendance_data():
    global result
    if request.method == "POST":
        result=[]
        subject_b=request.form['sub1']
        with open(os.path.join(file_path,subject_b+".csv"),'r') as csvFile2:
            csv_reader=csv.reader(csvFile2)
            for row in csv_reader:
                result.append(row)
            heading=result.pop(0)
    return render_template("print_attendance_data.html",heading=heading,result=result,subject_b=subject_b)





if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")