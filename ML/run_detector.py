import os
import torch
import cv2 as cv
import numpy as np

from blazepalm import PalmDetector
from handlandmarks import HandLandmarks

m = PalmDetector()
m.load_weights("./palmdetector.pth")
m.load_anchors("./anchors.npy")

hl = HandLandmarks()
hl.load_weights("./handlandmarks.pth")

cap = cv.VideoCapture(0)
while True:
	ret, frame = cap.read()
	hh, ww, _ = frame.shape
	ll = min(hh, ww)
	img = cv.resize(frame[:ll, :ll][:, ::-1], (256, 256))
	predictions = m.predict_on_image(img)
	for pred in predictions:
		for pp in pred:
			p = pp*ll
			# crop this image, pad it, run landmarks
			x = max(0, p[0])
			y = max(0, p[1])
			endx = min(ll, p[2])
			endy = min(ll, p[3])
			cropped_hand = frame[y:endy, x:endx]
			maxl = max(cropped_hand.shape[0], cropped_hand.shape[1])
			cropped_hand = np.pad(cropped_hand,
				( ((maxl-cropped_hand.shape[0])//2, (maxl-cropped_hand.shape[0]+1)//2), ((maxl-cropped_hand.shape[1])//2, (maxl-cropped_hand.shape[1]+1)//2) ),
				'constant')
			cropped_hand = cv.resize(cropped_hand, (256, 256))
			_, _, landmarks = hl(torch.from_numpy(cropped_hand).permute((2, 0, 1)).unsqueeze(0))
			print(landmarks)
			# cv.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)
			break

	cv.imshow('frame', frame)
	if cv.waitKey(1) == ord('q'):
		break

cap.release()
cv.destroyAllWindows()