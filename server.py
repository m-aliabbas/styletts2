from fastapi import FastAPI, HTTPException
from StyleTTSInference import StylTTSInference

# Initialize FastAPI app
app = FastAPI()
sttyle_tts_api = StylTTSInference()
# ...

@app.post("/synthesize/")
def synthesize(text: str, voice: str = '', lngsteps: int = 3):

    api_resp =  sttyle_tts_api.synthesize(text=text,voice='m-us-2',lngsteps=3)
    # Your synthesis logic here
    # ...

    # Return the synthesized speech
    return api_resp
