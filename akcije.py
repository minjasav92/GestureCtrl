import pyautogui
import cv2

# Mapiranje gestova na tastere na tastaturi
AKCIJE = {
    'otvorena_saka': lambda: pyautogui.press('right'),       # Sledeći slajd
    'pesnica':       lambda: pyautogui.press('left'),        # Prethodni slajd
    'palac_gore':    lambda: pyautogui.press('space'),       # Pauza / Play
    'dva_prsta':     lambda: pyautogui.hotkey('ctrl', '='), # Zoom in
    'jedan_prst':    lambda: None                            # Samo vizuelni overlay
}

def izvrsi(gest):
    if gest in AKCIJE:
        AKCIJE[gest]()
        print(f'Akcija izvrsena: {gest}')

def nacrtaj_overlay(frejm, gest):
    tekst = f"Gest: {gest}" if gest else "Gest: ---"
    # Crni pravougaonik na vrhu
    cv2.rectangle(frejm, (10, 10), (350, 70), (0, 0, 0), -1)
    # Zeleni tekst
    cv2.putText(frejm, tekst, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frejm