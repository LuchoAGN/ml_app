import uvicorn
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

#ML Pkg
import textstat 
import nltk
import string 
import re
import spacy
# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from wordfreq import word_frequency

#init app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Models
class TextProc(BaseModel):
    text: str = Field(...)

#Routes
@app.get('/')
async def index():
    return {"text": "Hello API Builders"}

@app.post('/score/')
async def score(text_data: TextProc = Body(...)):
    # Ícaro 
    inflessz = textstat.szigriszt_pazos(text_data.text)
    
    # Cálcula el número de caracteres
    num_char = len(text_data.text)
    
    # Función que calcula el número de palabras en un texto
    def count_words(string):
        words = string.split()
        return len(words)
    
    c_words = count_words(text_data.text)
    
    #Función que retorna la longitud promedio de una palabra 
    def avg_word_length(string):
        words = string.split()
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths)/len(words)
        return(avg_word_length)
    
    avg_w_len = avg_word_length(text_data.text)
    
    # spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
    # numero_de_oraciones = spanish_tokenizer.tokenize(text_data)
    
    return {"Result_Indice": (inflessz),
        "Numero_caracteres": num_char,
        "Numero_palabras": c_words,
        "Longitud_promedio_palabra": avg_w_len,
        # "Numero_Oraciones": numero_de_oraciones
    }


if __name__ == '___main___':
    uvicorn.run(app, host="127.0.0.1", port=8000)