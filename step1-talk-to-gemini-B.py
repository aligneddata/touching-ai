import os
from google import genai


def ask_gemini(user_ask: str) -> str:
    print(f"Question: {user_ask}")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{question}",
    )

    # print(response)
    return response.text

#
# main
#
while True:
    question = input("Please ask me anything (ENTER to exit): ") 
    if question == "":  # or: not question; or: len(question) == 0
        break
    
    response = ask_gemini(question)
    print(response)
