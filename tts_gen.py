from gtts import gTTS
import os
texts = ['WAKEUP. WAKE UP. WAKE UP. PLEASE. WAKE. UP.', 'You are sleepy. You need to have some COFFEE.', 'You seem to be Drowsing. Wake up right now.',
        'You are turning back. Please Look to the Front.', 'Please Look Forward. Please Look Forward. Please Look Forward.',
        'Concentrate on the Road. Not on your Phone.', 'You are with your phone for too long. Please put it down.','Lose your Phone Please. Keep your phone aside.']
        
for idx, t in enumerate(texts):    
    tts = gTTS(text=t, lang='en')
    print('Generating', 'aud'+str(idx)+'.mp3')
    tts.save('audio/aud'+str(idx)+'.mp3')