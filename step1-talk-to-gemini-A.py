def ask_gemini(user_ask: str) -> str:
    print(f"Question: {user_ask}")
    resp = "TBD"
    return resp

#
# main
#
while True:
    question = input("Please ask me anything (ENTER to exit): ") 
    if question == "":  # or: not question; or: len(question) == 0
        break
    
    response = ask_gemini(question)
    print(response)
