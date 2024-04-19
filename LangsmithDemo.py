#import openai
from openai import OpenAI
#from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import dotenv_values

# config
config = dotenv_values(".env")
api_key = config["OPENAI_API_KEY"]
print("API Key:")
print(api_key)
print("\n")

# Auto-trace LLM calls in-context
#client = wrap_openai(OpenAI(api_key=api_key))
client = OpenAI(api_key=api_key)
print("Client:")
print(client)
print("\n")

# auto trace with func
@traceable
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content

resp = pipeline("Hello, world!")
print("Resp:")
print(resp)
print("\n")
