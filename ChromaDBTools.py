import os, sys, logging
import chromadb
import numpy as np
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
from chromadb.utils import embedding_functions

import Utils

CHUNK_SIZE = 2048

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#                 model_name="text-embedding-ada-002"
#             )

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8', 
                    level=os.getenv('DEBUG_LEVEL', 'INFO'))


class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    EMBEDDING_MODEL_ID = "models/embedding-001" # "models/text-embedding-004"
    title = "Custom query"
    logging.debug("Embedding req >>>>>> [%s]. Size [%d]" % (input, len(input)))
    response = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=input,
        config=types.EmbedContentConfig(
          task_type="retrieval_document",  # SEMANTIC_SIMILARITY
          #title=title
        )
    )

    embeds = response.embeddings[0].values
    logging.debug("Embedded resp <<<<<< %s. Size [%d] " % (embeds, len(embeds)))
    return embeds    

  def encode_question(self, input):
    return self.__call__(input)    
      

class ChromaDBTools:
    def __init__(self):
        self.chroma_client = chromadb.EphemeralClient()
        
    def load_from_file(self, index_name: str, filename: str):
        db = self._get_collection_by_name(index_name)
        contents = Utils.read_file_to_string(filename)
        chunks = Utils.split_into_chunks(contents, chunk_size=CHUNK_SIZE)
        for i, d in enumerate(chunks):
            db.add(documents=d, ids=str(i))
        return db

    def similarity_search(self, index_name: str, query: str, k=4):
        db = self._get_collection_by_name(index_name)
        passages = db.query(query_texts=[query], n_results=k)['documents']
        if passages and passages[0]:
            return passages[0]
        else:
            return None

    def _get_collection_by_name(self, index_name):
        existing_collections = self.chroma_client.list_collections()
        for collection in existing_collections:
            if collection.name == index_name:
                return collection
        return self._create_chroma_db(index_name)

    def _create_chroma_db(self, name):            
        db = self.chroma_client.create_collection(
            name=name,
            embedding_function=GeminiEmbeddingFunction()
        )
        return db

vdb = ChromaDBTools()
vdb.load_from_file("chapter2", "data\\chapter2.txt")
context = vdb.similarity_search("chapter2", "How to use break pad")
logging.debug(context)

