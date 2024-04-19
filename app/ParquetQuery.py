import os
from datetime import datetime
from datasets import load_dataset
from dotenv import dotenv_values
import json
import ray
import requests
import pickle
import pandas as pd
import numpy as np

from pinecone import Pinecone as PineconeClient
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from dotenv import dotenv_values
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Pinecone 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import requests
import json
import ray

config = dotenv_values(".env")
env_key = config["PINE_CONE_ENV_KEY"]
api_key = config["PINE_CONE_API_KEY"]
openai_api_key=config["OPENAI_API_KEY"]
cohere_api_key = config["COHERE_API_KEY"]
print(f"Cohere api key = {cohere_api_key}")
print(f"OpenAI api key = {openai_api_key}")

pineCone = PineconeClient(
    api_key=api_key
)

# create the index, vectorstore and retriever
index_name = "cohere-wikipedia" 
embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = vectorstore.as_retriever()

print("Retriever:")
print(retriever)
print("\n")

def fetch_wikipedia_page(id):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&pageids={id}"
    response = requests.get(url)
    data = response.json()
    page_content = list(data['query']['pages'].values())[0]['extract']
    return page_content

def fetch_url(x):
    print(f"Fetch url...")
    print("x['context']:")
    print(x['context'])
    print("\n")
    print("x['question']:")
    print(x['question'])
    print("\n")
    urls = [doc.metadata['url'] for doc in x['context']] 
    ids = [url.split('=')[-1] for url in urls]
    contents = [fetch_wikipedia_page(id)[:32000] for id in ids]
    return {"context": contents, "question": x["question"]}

# RAG Prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG chain
#model = ChatOpenAI(model="gpt-4-32k", openai_api_key=openai_api_key)
model = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=openai_api_key)
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | RunnableLambda(fetch_url)
    | prompt
    | model
    | StrOutputParser()
)
chain_result = chain.invoke("What is film noir?")

print("Chain result output:")
print(chain_result)
print("\n")

