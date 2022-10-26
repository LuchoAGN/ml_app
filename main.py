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
import pandas as pd 
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

@app.post("/metricas/")
async def score(text_data: TextProc =  Body(...)):
    # Ícaro 
    inflessz = textstat.szigriszt_pazos(text_data.text)
    level_inflesz = ""
    color_level = ""
    
    if inflessz <= 40:
        level_inflesz = "Muy difícil"
        color_level = "#DD4B39"
    elif inflessz > 40 and inflessz <= 55:
        level_inflesz = "Algo difícil"
        color_level = "#DD4B39"
    elif inflessz > 55 and inflessz <= 65:
        level_inflesz = "Normal"
        color_level = "#F39C12"
    elif inflessz > 66 and inflessz <= 80:
        level_inflesz = "Algo fácil"
        color_level = "#00A65A"
    else: 
        level_inflesz = "Muy fácil"
        color_level = "#00A65A"
        
    # Cálcula el número de caracteres
    num_char = len(text_data.text)
    
    # Función que calcula el número de palabras en un texto
    def count_words(string):
        words = string.split()
        return len(words)
    
    c_words = count_words(text_data.text)
    
    # Promedio de palabras por oración 
    def avg_sentence_length(x):
            
        sentences = nltk.sent_tokenize(x)
        words_in_sentence_length = [count_words(sentences) for sentences in sentences]    
        avg_sentence_length = sum(words_in_sentence_length) / len(words_in_sentence_length)
        return(avg_sentence_length)

    avg_w_len = avg_sentence_length(text_data.text)
    
    #Oración mas larga 
    senteces_tokens = nltk.sent_tokenize(text_data.text)
    oracion_mas_larga = max(senteces_tokens, key = len)
    
    #Número de oraciones en el texto
    senteces_tokens = nltk.sent_tokenize(text_data.text)
    numero_oraciones = len(senteces_tokens)
    
    # Número de palabras en formato capital
    Capital_words = re.findall(r'(?<!^)(?<!\. )[A-Z][a-z]+', text_data.text)
    num_palabras_cap = len(Capital_words)
    
    ## Numero de signos de puntuación en el texto 
    sign_punt = r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]"
    Punt_sing = len(re.findall(sign_punt, text_data.text))

    # Top de palabras dificiles en el text --> Palabras complejas
    def word_freq(x):
        words = x.split()
        word_freqc = [word_frequency(word, 'es') for word in words]
        df = pd.DataFrame(list(zip(words, word_freqc)), columns =['Word', 'Freq'])
        return(df)

    dfWord = word_freq(text_data.text).sort_values(by = ['Freq'])
    palabras_dificiles =  dfWord['Word'].head(n=5)
    
    #PDD - Etiquetado
    nlp = spacy.load("es_core_news_sm")

    def pos_tag(text):
        
        doc = nlp(text)
        pos = [(token.text, token.pos_) for token in doc]
        df = pd.DataFrame(pos, columns = ['Etiqueta', 'Frecuencia'])
        df_counts = df['Frecuencia'].value_counts()
        return df_counts

    print(pos_tag(text_data.text).to_frame())
    

    def ner_tag (text):

        doc = nlp(text)
        ne = [(ent.text, ent.label_) for ent in doc.ents]
        df = pd.DataFrame(ne, columns = ['Texto', 'Etiqueta'])
        df_counts = df.groupby(['Etiqueta', 'Texto'])['Texto'].count()
        
        return df_counts

    print(ner_tag(text_data.text).to_frame())
    
    return {
        "Result_Indice": {
            "IFSZ": (inflessz),
            "level_inflesz": level_inflesz,
            "color_level": color_level
            },
        "Numero_caracteres": num_char,
        "Numero_palabras": c_words,
        "promedio_palabras": avg_w_len,
        "oracion_mas_larga": oracion_mas_larga,
        "numero_oraciones": numero_oraciones,
        "num_palabras_cap": num_palabras_cap,
        "num_signos_punt": Punt_sing,
        "palabras_dificiles": palabras_dificiles
    }


if __name__ == '___main___':
    uvicorn.run(app, host="127.0.0.1", port=8000)