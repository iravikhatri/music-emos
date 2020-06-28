import os
import time
import subprocess
import random
import platform

import numpy as np
import cv2

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image


hff_path = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(hff_path)

model_path = os.path.join(os.getcwd(), 'emotion_little_vgg.h5')
classifier = load_model(model_path)

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

timeout = time.time() + 15 

while True:
	_, frame = cap.read()
	labels = []
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)

	for (x, y, w, h) in faces:

		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

		if np.sum([roi_gray]) != 0:
			roi = roi_gray.astype('float')/255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			predicts = classifier.predict(roi)[0]
			label = class_labels[predicts.argmax()]
			label_position = (x, y)

			cv2.putText(frame, label, label_position,
						cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

		else:
			cv2.putText(frame, 'No Face Found', (20, 60),
						cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

	cv2.imshow('Emotion Detector', frame)

	if time.time() > timeout:
		song_folder = os.path.join(os.getcwd(), f"Songs/{label}")
		song = random.choice(os.listdir(song_folder))

		if platform.system() == "Windows":
			music_player = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
			subprocess.call([mp, f'{song_folder}/{song}'])
		else:
			os.system(f'vlc {song_folder}/{song}')
			break

	if cv2.waitKey(30) & 0xff == 27: # The Esc key
		break

cap.release()
cv2.destroyAllWindows()


			
			