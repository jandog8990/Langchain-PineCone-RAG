from pinecone import Pinecone as PineconeClient
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
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
pc_index_name = config["INDEX_NAME"]
print(f"openai key = {openai_api_key}")
print(f"cohere key = {cohere_api_key}")

# setup the embeddings and pinecone index
pc = PineconeGRPC()
pinecone = PineconeClient(api_key=api_key)
embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
vectorstore = Pinecone.from_existing_index(index_name=pc_index_name,
                                           embedding=embeddings)
retriever = vectorstore.as_retriever()

def fetch_wikipedia_page(id):
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&pageids={id}"
    response = requests.get(url)
    data = response.json()
    page_content = list(data['query']['pages'].values())[0]['extract']
    return page_content

def fetch_url(x):
    urls = [doc.metadata['url'] for doc in x['context']]
    ids = [url.split('=')[-1] for url in urls]
    contents = [fetch_wikipedia_page(id)[:32000] for id in ids]    
    return {"context": contents, "question": x["question"]}


# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", openai_api_key=openai_api_key)

#    | RunnableLambda(fetch_url)  # Add this line for granularity
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


