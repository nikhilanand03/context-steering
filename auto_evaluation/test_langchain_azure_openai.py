import os
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    openai_api_version="2024-02-01",
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)

text = "this is a test document"

query_result = embeddings.embed_query(text)
print(len(query_result))

doc_result = embeddings.embed_documents([text])
print(doc_result[0][:5],len(doc_result),len(doc_result[0]))