import os, sys
from google import genai
import settings

def ask_gemini(client, myfile, user_ask: str) -> str:   
    print(f"Question: {user_ask}")
    response = client.models.generate_content(
        model=settings.MODEL_NAME,
        contents=[myfile, user_ask]
    )
    return response.text


#
# main
#
filename = sys.argv[1]
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))    
    myfile = client.files.upload(file=filename)   
except Exception as e:
    print(e)
    sys.exit(1)
 

while True:
    question = input("Please ask me anything (ENTER to exit): ") 
    if question == "":  # or: not question; or: len(question) == 0
        break
    
    ADDITIONAL_REQUIREMENTS = ""
    response = ask_gemini(client, myfile, ADDITIONAL_REQUIREMENTS + question)
    print(response)
