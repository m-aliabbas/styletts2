import styletts2importable
import ljspeechimportable
import torch
import os
from txtsplit import txtsplit
import numpy as np
import pickle
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

class StylTTSInference:
    def __init__(self,voiceslist=['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']) -> None:
        self.voicelist = voiceslist
        self.voices = {}
        self.compute_voice_style()

    def compute_voice_style(self):
        print('---- Computing Style ----')
        print(self.voicelist)
        for v in self.voicelist:
            print(f'---- Running voice: voices/{v}.wav ----')
            self.voices[v] = styletts2importable.compute_style(f'voices/{v}.wav')

    def synthesize(self,text, voice='m-us-2', lngsteps=3):
        '''
        text(str): Text to narrate
        voice(str): Text to Narrate
        lngsteps(int): Duffision Step
        '''
        if text.strip() == "":
            return {'status':0,'content':{'sr':0,'data':None},'msg':'Error! Please input some more data'}
        if len(text) > 50000:
            return {'status':0,'content':{'sr':0,'data':None},'msg':'Error! Should be less than 50k characters'}
        texts = txtsplit(text)
        v = voice.lower()
        audios = []
        for t in texts:
            temp_ = styletts2importable.inference(t, self.voices[v], alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1)
            # print(type(temp_.tostring()))
            # print(styletts2importable.inference(t, self.voices[v], alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1))
            audios.append(temp_.tolist())
        data = (24000, audios)
        return {'status':1,'content':{'sr':data[0],'data':data[1]},'msg':'Success'}
    
    def get_voice_list(self):
        return {'voice_list':self.voicelist}
