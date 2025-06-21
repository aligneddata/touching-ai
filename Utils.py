import os, sys, logging
import chromadb
import numpy as np
from google import genai
from google.genai import types
from pinecone import Pinecone
import os
import itertools

def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

CHUNK_SIZE = 2048

def read_file_to_string(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as file:
        return file.read()

def split_into_chunks(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""
        You are a helpful and informative bot that answers questions using
        text from the reference passage included below.
        Be sure to respond in a complete sentence, being comprehensive,
        including all relevant background information.
        However, you are talking to a non-technical audience, so be sure to
        break down complicated concepts and strike a friendly
        and converstional tone. If the passage is irrelevant to the answer,
        you may ignore it, and say I DO NOT KNOW.
        Reference passage: '{relevant_passage}'
        Question: '{query}'

        Answer:
    """).format(query=query, relevant_passage=escaped)
    return prompt

def ask_gemini(enriched_ask: str) -> str:
    print(f"Question: {enriched_ask}")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{enriched_ask}",
    )
    return response.text
