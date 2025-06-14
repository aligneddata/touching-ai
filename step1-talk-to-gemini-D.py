import os
from google import genai
from google.genai import types


def ask_gemini(user_ask: str) -> str:
    print(f"Question: {user_ask}")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{user_ask}",
        config=types.GenerateContentConfig(
            temperature=2.0  # try both 0 and 2.0
        )
    )

    for item in response:
        print("Response item: ", item)
    return response.text

#
# main
#
while True:
    question = input("Please ask me anything (ENTER to exit): ") 
    if question == "":  # or: not question; or: len(question) == 0
        break
    
    ADDITIONAL_REQUIREMENTS = "ANSWER THIS QUESTION TWICE: ONE IN FORMAL WAY AND ONE IN hilarious WAY: "
    response = ask_gemini(ADDITIONAL_REQUIREMENTS + question)
    print(response)
