import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Kod iz Nedelje 3 (Prepoznaj sa pragom i Stabilan gest) ---
def prepoznaj_sa_pragom(vektor, prag=0.7):
    proba = model.predict_proba(vektor.reshape(1, -1))[0]
    if proba.max() < prag:
        return None
    indeks = proba.argmax()
    return model.classes_[indeks]

from collections import deque, Counter
BUFFER = deque(maxlen=10)

def stabilan_gest(novi_gest):
    BUFFER.append(novi_gest)
    if len(BUFFER) < 10:
        return None
    najcesci, broj = Counter(BUFFER).most_common(1)[0]
    if broj >= 7:
        BUFFER.clear()
        return najcesci
    return None

def landmarks_na_vektor(rezultat):
    lm = rezultat.multi_hand_landmarks[0].landmark
    return np.array([[p.x, p.y, p.z] for p in lm]).flatten()

# --- Korak 1 — Mapiranje gestova na akcije ---
AKCIJE = {
    'otvorena_saka': lambda: pyautogui.press('right'), # sledeći slajd
    'pesnica': lambda: pyautogui.press('left'), # prethodni slajd
    'palac_gore': lambda: pyautogui.press('space'), # pauza / play
    'dva_prsta': lambda: pyautogui.hotkey('ctrl', '='), # zoom in
    'jedan_prst': lambda: None, # laser pointer — samo overlay
}

def izvrsi(gest):
    if gest in AKCIJE:
        AKCIJE[gest]()
        print(f'Akcija izvrsena: {gest}')

# --- Korak 2 — Overlay UI ---
def nacrtaj_overlay(frejm, gest):
    tekst = f"Gest: {gest}" if gest else "Gest: ---"
    cv2.rectangle(frejm, (10, 10), (350, 70), (0, 0, 0), -1)
    cv2.putText(frejm, tekst, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frejm

# Inicijalizacija modela i kamere
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

# --- Korak 3 — Finalna petlja ---
while True:
    ok, frejm = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frejm, cv2.COLOR_BGR2RGB)
    rez = mp_hands.process(rgb)
    gest = None
    
    if rez.multi_hand_landmarks:
        vektor = landmarks_na_vektor(rez)
    ### START CODE HERE ###
        gest = prepoznaj_sa_pragom(vektor)
        finalni = stabilan_gest(gest)
        if finalni:
            izvrsi(finalni)
    
    # PROMENJENO: Prosleđujemo 'gest' umesto 'finalni' da bi tekst pratio ruku u realnom vremenu
    frejm = nacrtaj_overlay(frejm, gest)
    ### END CODE HERE ###
        
    cv2.imshow("Kontrola Prezentacije - Live", frejm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()