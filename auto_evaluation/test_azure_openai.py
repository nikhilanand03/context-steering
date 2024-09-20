import os
from openai import AzureOpenAI
    
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
deployment_name='gpt-4o' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 
    
# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a tagline for an ice cream shop. '
response = client.chat.completions.create(model=deployment_name, 
    messages=[
            {"role": "system", "content": "Assistant is a large language model trainOpenAIed by ."},
            {"role": "user", "content": "Who were the founders of Microsoft?"}
        ],
    max_tokens=10)
# print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)