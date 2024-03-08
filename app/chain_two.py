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

# connecto to pinecone api
config = dotenv_values(".env")
env_key = config["PINE_CONE_ENV_KEY"]
api_key = config["PINE_CONE_API_KEY"]
openai_api_key=config["OPENAI_API_KEY"]
cohere_api_key = config["COHERE_API_KEY"]
#pc_index = config["INDEX_NAME"]
print(f"openai api_key= {openai_api_key}")
print(f"pinecone env_key = {env_key}")
print(f"pinecone api_key = {api_key}")
print("\n")

# pinecone client and embeddings
pineCone = PineconeClient(
    api_key=api_key
)

# start to load the data
url = "https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/default/train"
response = requests.get(url)
input_files = json.loads(response.content)
columns = ['id', 'title', 'text', 'url', 'emb']
ds = ray.data.read_parquet(input_files, columns=columns) 
print("Dataset:")
print(ds)
print("\n")

# create the index for wikipedia data
"""
indexes = pc.list_indexes().indexes
print("Indexes:")
print(indexes)
print("\n")
"""

"""
embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
print("Embeddings:")
print(embeddings)
print("\n")
"""

# get the index from existing srcs
"""
vectorstore = Pinecone.from_existing_index(index_name=pc_index,
        embedding=embeddings)
retriever = vectorstore.as_retriever()
print("Vector store:")
print(vectorstore)
print("\n")

def fetch_wiki_page(id):
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
"""

# RAG prompt
template = """Answer the question based only on the following context:
    {context}
    Question: {question}
"""
#prompt = ChatPromptTemplate.from_template(template)

# RAG model
#model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", openai_api_key=openai_api_key)

"""
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | RunnableLambda(fetch_url)
    | prompt
    | model
    | StrOutputParser()
)
"""
