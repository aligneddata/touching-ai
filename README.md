# Purpose
An exercise that uses Python to introduce basic AI concepts within 8-10 hours.

# Pre-requisite
* Python: Junior to intermediate level. Comfortable with basic data types such as list, file operations, and internet http calls.

# Setup
* Conda based Python: Download from https://www.anaconda.com/download and install it in your PC.
* Git tools: Download from https://git-scm.com/downloads/win and install it in your PC.
* Register a free account at https://github.com/
* Register a Google account and request a Gemini API key: https://aistudio.google.com/app/apikey. Save the key in a safe place.
* Register a free account at [pinecone.io](https://www.pinecone.io/) and request a free API key. Save the key in a safe place.
* IDE: Install the Spyder components from the previously downloaded Anaconda suite, or download from https://code.visualstudio.com/, or choose any other IDE of your preference.
* Setup conda environment for this exercise.
<pre>
conda create -n touching-ai python=3.11
conda activate touching-ai
pip install -q -U google-genai
</pre>

# Step 1. Talk to Gemini by API
Purpose: Create a Python program that takes user questions, forwards to Gemini via API, and prints out answers on screen.
* Configurations:
  - In Windows, search and click "Edit Environment Variables for Your Account".
  - In the opened dialog, click "NEW" to create a variable named GEMINI_API_KEY and fill the variable value field using the Gemini API key string that registered during the Setup stage.
  - Close existing Anaconda DOS window, if any. 
  - Open a new DOS window. Run "conda activate touching-ai" if the current environment is not.
  - Run command "pip install -q -U google-genai". Run "python step1-verify-gemini.py" to verify.
* Exercise:
  - Create a Python program that has an indefinite loop. Inside the loop, the program takes user input.
  - Create a function that can make API calls to Gemini and return textual response.
  - Print out the answer to user screen.
  - Go back to the function, print out the full Gemini response, understand the returned data.
  - Continue the loop till user inputs an empty screen.
  - Practise controlling API calling by adding additional information into prompt.
  - Practise controlling API calling by enforcing parameters. https://ai.google.dev/gemini-api/docs/prompting-strategies#model-parameters
    + https://ai.google.dev/api/generate-content#v1beta.GenerationConfig
    + Try temperature [0, 2.0]
  - Question to ask: dog or cat, which is smarter?
* More:
  - Read: https://ai.google.dev/gemini-api/docs/structured-output
* Challenge:
  - Create a program that takes a user's question, and limit the answer to 50 tokens. Try both ways: 1. Using additional requirement in prompt. 2. Using maxOutputTokens. See which is better.

# Step 2. GenAI: Vector, embeddings, vector database
Purpose: Learn the concept of vector and embeddings and their use.
* Configuration
<pre>
pip install -U -q "google-genai>=1.0.0"
pip install -q chromadb
pip install pandas
</pre>
* Exercise:
  - vector-similarity: Vector, Euclidean distance, Cosine similarity
  - embeddings: Talk to Google API to encode text to embeddings.
  - talk-to-pinecone
  - driver
     * Utils
     * VectoDbPinecone
* Challange:
  - Download an interesting book in text format, history, geo, or even fiction. Break it down to smaller chunks. Upload to an index in Pinecone.
  - Make a program that accepts user's inquries. 
  - Answer question: anything more we can do when the answer is "I DO NOT KNOW"?

# Step 3. Google API for image/video/audio files
Purpose: Use Google genai to upload a file to Gemini and ask questions related to the file. Demo image and video file.

# Step 4. Pytorch. Neural Network
 * Concepts/Exercises:
    - Pytorch tensor
    - Gradient and backward()
    - nn.Module and custom forward()
    - Simple 1 layer network
    - Dataset, split, dataloader, and general epoch structure
    - Save and restore work
    - Two layer network
  * Configuration
  <pre>
  conda activate touching-ai
  pip3 install torch torchvision
  </pre>