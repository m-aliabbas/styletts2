from fastapi import FastAPI, HTTPException
from StyleTTSInference import StylTTSInference
from pydantic import BaseModel
import numpy as np
# Pydantic model
class SynthesizeRequest(BaseModel):
    text: str
    voice: str = 'm-us-2'
    lngsteps: int = 3

# Initialize FastAPI app
app = FastAPI()
sttyle_tts_api = StylTTSInference()

@app.post("/synthesize/")
def synthesize(request: SynthesizeRequest):
    api_resp = sttyle_tts_api.synthesize(text=request.text, voice=request.voice, lngsteps=request.lngsteps)
    print(api_resp)
    return api_resp

