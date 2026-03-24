from google import genai
from google.genai.types import HttpOptions

# 1. Setup with the stable v1 API version
client = genai.Client(
    api_key='key goes here',
    http_options=HttpOptions(api_version='v1')
)

# 2. Use the current 2026 stable model: gemini-2.5-flash
response = client.models.generate_content(
    model='gemini-2.5-flash', 
    contents="Hello from Romania! Is the stable API working now?"
)

# 3. Print the result
print(response.text)