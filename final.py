import cv2
import mediapipe as mp
import time
from collections import deque
import pyttsx3
import speech_recognition as sr
from openai import OpenAI
import subprocess
import threading
import keyboard
import sys
import os


STILL_THRESHOLD = 0.03         
STILL_DURATION = 2             
WAVE_DEQUE_LEN = 8             
WAVE_DIRECTION_CHANGES_REQUIRED = 2  
WAVE_MOVEMENT_THRESHOLD = 0.15 

API_KEY = "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"  #
BASE_URL = "https://openrouter.ai/api/v1"

engine = pyttsx3.init()

movement_initiated = False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

hand_x_positions = deque(maxlen=WAVE_DEQUE_LEN)
last_wave_time = 0
still_start_time = None
ready_for_wave = False
previous_wrist_x = None

reference_area = None  

def get_chatbot_response(query):
    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
        )
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",   
                "X-Title": "MyCollegeProject"           
            },
            extra_body={},
            model="cognitivecomputations/dolphin3.0-mistral-24b:free",
            messages=[
                {"role": "user", "content": query}
            ]
        )
        answer = completion.choices[0].message.content.strip()
        print(answer)
        return answer
    except Exception as e:
        print("Error in chatbot response:", e)
        return "I'm sorry, I could not get a response at the moment."

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def conversation_mode():
    
    global cap, movement_initiated

    greeting = "Hello, how may I help you?"
    speak_text(greeting)
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your question...")
        try:
            audio = recognizer.listen(source, phrase_time_limit=5)
            user_query = recognizer.recognize_google(audio)
            print("User query:", user_query)
        except Exception as e:
            print("Speech recognition error:", e)
            speak_text("I did not catch that. Please try again later.")
            return

    
    if user_query.strip().lower() == "initialise movement":
        cap.release()
        cv2.destroyAllWindows()
        subprocess.Popen([sys.executable, "hand detection.py"])
        keyboard.unhook_all()
        os._exit(0)
    else:
        answer = get_chatbot_response(user_query)
        print("Chatbot answer:", answer)
    
    cancel = False
    tts_thread = threading.Thread(target=speak_text, args=(answer,))
    tts_thread.daemon = True
    tts_thread.start()
    
    
    while tts_thread.is_alive():
        if keyboard.is_pressed('c'):
            engine.stop()  
            cancel = True
            break
        time.sleep(0.01)
    
    if cancel:
        keyboard.unhook_all()
        cap.release()
        cv2.destroyAllWindows()
        subprocess.Popen([sys.executable, "hand detection.py"])
        os._exit(0)
    else:
        tts_thread.join()

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_time = time.time()

    
    pose_results = pose.process(frame_rgb)
    hand_results = hands.process(frame_rgb)

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        x_coords = [int(landmark.x * w) for landmark in landmarks]
        y_coords = [int(landmark.y * h) for landmark in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        bbox_area = (max_x - min_x) * (max_y - min_y)

        if reference_area is None:
            reference_area = bbox_area

        if bbox_area < 0.5 * reference_area:
            cv2.putText(frame, "Human far (Move closer)", (min_x, min_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Human near", (min_x, min_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x = wrist.x

            
            if not ready_for_wave:
                if previous_wrist_x is not None:
                    if abs(wrist_x - previous_wrist_x) < STILL_THRESHOLD:
                        if still_start_time is None:
                            still_start_time = current_time
                        elif current_time - still_start_time >= STILL_DURATION:
                            ready_for_wave = True
                            hand_x_positions.clear()
                    else:
                        still_start_time = None
                        ready_for_wave = False
                else:
                    still_start_time = current_time

            previous_wrist_x = wrist_x

            
            if ready_for_wave:
                hand_x_positions.append(wrist_x)
                if len(hand_x_positions) == hand_x_positions.maxlen:
                    diffs = [hand_x_positions[i+1] - hand_x_positions[i] for i in range(len(hand_x_positions)-1)]
                    direction_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
                    
                    if (direction_changes >= WAVE_DIRECTION_CHANGES_REQUIRED and 
                        (max(hand_x_positions) - min(hand_x_positions)) > WAVE_MOVEMENT_THRESHOLD):
                        if current_time - last_wave_time > 3:
                            last_wave_time = current_time
                            cv2.putText(frame, "Wave detected! Initiating conversation...", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                            conversation_mode()
                        ready_for_wave = False
                        still_start_time = None
                        hand_x_positions.clear()

    cv2.imshow('Gesture, Spatial Awareness & AI Conversation', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


    if movement_initiated:
        break

cap.release()
cv2.destroyAllWindows()
