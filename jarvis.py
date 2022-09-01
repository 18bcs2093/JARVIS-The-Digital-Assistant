#project jarvis
import speech_recognition as sr
import datetime
import wikipedia
import pyttsx3
import webbrowser
import os
import cv2
import numpy as np
import pyautogui as p


#text to speech

engine = pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
# print(voices)
engine.setProperty(voices,voices[1].id)

def speak(audio): #share audio var which contain text
    engine.say(audio)
    engine.runAndWait()

def wish():
    hour=int(datetime.datetime.now().hour)
    if hour >= 0 and hour<12:
        speak("good morning sir i am your virtual assistent jarvis")
    elif hour>=12 and hour<18:
        speak("good afternoon sir i am your virtual assistent jarvis")
    else:
        speak("good night sir i am your virtual assistent jarvis")


#now convert audio to text

def takecom():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.......")
        audio=r.listen(source)
    try:
        print("Recognising.")
        text=r.recognize_google(audio,language="en-in")
        print(text)
    except Exception:
        speak("error...")
        print("Cannat recognise")
        return "none"
    return text

#for main function
def TaskExecution():
    p.press('esc')
    print("face recognision successfull")
    speak("face recognision successfull")
    speak('welcome master ankit')
    # wish()
    while True:
        query = takecom().lower()

        if "wikipedia" in query:
            speak("searching details....wait")
            query.replace("wikipedia","")
            results=wikipedia.summary(query,sentences=2)
            print(results)
            speak(results)
        elif 'open youtube' in query or 'open youtube online' in query:
            webbrowser.open("www.youtube.com")
            speak("opening youtube")
        elif 'open google' in query:
            webbrowser.open("www.google.co.in")
            speak("opening google")
        elif 'music from pc' in query or 'music' in query:
            speak("ok i am playing music")
            music_dir="./music"
            musics=os.listdir(music_dir)
            os.startfile(os.path.join(music_dir,musics[0]))
        elif 'video from pc' in query or 'video' in query:
            speak("ok i am playing videos")
            video_dir="./video"
            videos=os.listdir(video_dir)
            os.startfile(os.path.join(video_dir,videos[0]))
        elif 'good bye' in query:
            speak("good bye")
            exit()
        elif 'shutdown' in query:
            speak("shutting down")
            os.system('shutdown -s')



if __name__=="__main__":
    recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms
    recognizer.read('trainer/trainer.yml')   #load trained model
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath) #initializing haar cascade for object detection approach

    font = cv2.FONT_HERSHEY_SIMPLEX #denotes the font type


    id = 2 #number of persons you want to Recognize


    names = ['','ankit']  #names, leave first empty bcz counter starts from 0


    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cv2.CAP_DSHOW to remove warning
    cam.set(3, 640) # set video FrameWidht
    cam.set(4, 480) # set video FrameHeight

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)


    while True:

        ret, img =cam.read() #read the frames using the above created object

        converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #The function converts an input image from one color space to another

        faces = faceCascade.detectMultiScale( 
            converted_image,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #used to draw a rectangle on any image

            id, accuracy = recognizer.predict(converted_image[y:y+h,x:x+w]) #to predict on every single image

            # Check if accuracy is less them 100 ==> "0" is perfect match 
            if (accuracy < 100):
                id = names[id]
                accuracy = "  {0}%".format(round(100 - accuracy))
                print("Recognising face....")
                speak("recognising face")
                cam.release()
                TaskExecution()
  
            else:
                id = "unknown"
                accuracy = "  {0}%".format(round(100 - accuracy))
                print("Recognising face....")
                speak("recognising face")
                speak("User authentication is failed")
                cam.release()
                break
        
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(accuracy), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
        cv2.imshow('camera',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("Thanks for using this program, have a good day.")
    # cam.release()
    cv2.destroyAllWindows()


