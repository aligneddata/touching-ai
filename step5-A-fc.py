# original url: https://ai.google.dev/gemini-api/docs/function-calling

import pprint
import settings
# Step 1: Define a function declaration

# Define a function that the model can call to control smart lights
set_light_values_declaration = {
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

# This is the actual function that would be called based on the model's suggestion
def set_light_values(brightness: int, color_temp: str) -> dict[str, int | str]:
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    print(f"Function being actually called: Setting brightness to {brightness} and color temperature to {color_temp}")
    return {"brightness": brightness, "colorTemperature": color_temp}


# Step 2: Call the model with function declarations
from google import genai
from google.genai import types

# Configure the client and tools
client = genai.Client()
tools = types.Tool(function_declarations=[set_light_values_declaration])
config = types.GenerateContentConfig(tools=[tools])

# Define user prompt
contents = [
    types.Content(
        role="user", parts=[types.Part(text="Turn the lights down to a romantic level")]
    )
]

# Send request with function declarations
response = client.models.generate_content(
    model=settings.MODEL_NAME,
    contents=contents,
    config=config,
)

print(response.candidates[0].content.parts[0].function_call)
pprint.pprint(response, indent=4)
# It is not always the model will suggest a function call. E.g. a model may responsd with a question back to the user.
# For example: I can help with that, but I need a little more information. What brightness level would you like 
# (0-100), and what color temperature would you prefer (daylight, cool, or warm)? no function call suggested by 
# the model. Exiting.
if not response.candidates[0].content.parts[0].function_call:
    print(response.candidates[0].content.parts[0].text, "no function call suggested by the model. Exiting.")
    exit(0)

# thinking budget
import base64
# After receiving a response from a model with thinking enabled
# response = client.models.generate_content(...)

# The signature is attached to the response part containing the function call
part = response.candidates[0].content.parts[0]
if part.thought_signature:
  print("thought_signature:", base64.b64encode(part.thought_signature).decode("utf-8"))
  

# Step 3: Execute set_light_values function code if the model decided to call it
# Extract tool call details, it may not be in the first part.
tool_call = response.candidates[0].content.parts[0].function_call

if tool_call.name == "set_light_values":
    result = set_light_values(**tool_call.args)
    print(f"Function execution result: {result}")


# Step 4: Create user friendly response with function result and call the model again
# Create a function response part
function_response_part = types.Part.from_function_response(
    name=tool_call.name,
    response={"result": result},
)

# Append function call and result of the function execution to contents
contents.append(response.candidates[0].content) # Append the content from the model's response.
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

final_response = client.models.generate_content(
    model=settings.MODEL_NAME,
    config=config,
    contents=contents,
)

print(final_response.text)
pprint.pprint(final_response, indent=4)


