import os
from PIL import Image
from google import genai
import settings

def ask_gemini(image: str, user_ask: str) -> str:
    print(f"Question: {user_ask}")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))    
    response = client.models.generate_content(
        model=settings.MODEL_NAME,
        contents=[image, user_ask]
    )
    return response.text


#
# main
#
image = Image.open("data/cups.jpg")
while True:
    question = input("Please ask me anything (ENTER to exit): ") 
    if question == "":  # or: not question; or: len(question) == 0
        break
    
    ADDITIONAL_REQUIREMENTS = ""  # "ANSWER THIS QUESTION TWICE: ONE IN FORMAL WAY AND ONE IN hilarious WAY: "
    response = ask_gemini(image, ADDITIONAL_REQUIREMENTS + question)
    print(response)
