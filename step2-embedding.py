import os
import sys
import logging
from google import genai

# https://ai.google.dev/gemini-api/docs/models#text-embedding
# input token limit; output dimension size

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8', 
                    level=os.getenv('DEBUG_LEVEL', 'DEBUG'))

CHUNK_SIZE = 2048
MODEL = "models/embedding-001" # "models/text-embedding-004", "gemini-embedding-exp-03-07"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
def encode(text: str):
    logging.debug("Embedding req >>>>>> [%s]" % text[0:CHUNK_SIZE])
    result = client.models.embed_content(
            model=MODEL,  
            contents=text[0:CHUNK_SIZE])
    logging.debug("Embedded resp <<<<<< %s " % result)
    if result and result.embeddings and result.embeddings[0]:
        return result.embeddings[0].values
    else:
        return None

def decode(embeddings):
    pass


message = sys.argv[1]
embeds = encode(message)
print(embeds, len(embeds))

'''
Embeddings:
Vectors designed to represent words, sentences, documents, or visual/audio features.
Lower-dimensions: handreds to thousands
Context aware
'''

