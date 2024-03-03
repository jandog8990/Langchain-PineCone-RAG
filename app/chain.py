from pinecone import Pinecone as PineconeClient
from dotenv import dotenv_values
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Pinecone 
import requests

# connecto to pinecone api
config = dotenv_values(".env")
env_key = config["PINE_CONE_ENV_KEY"]
api_key = config["PINE_CONE_API_KEY"]
pc_index = config["INDEX_NAME"]
print(f"env_key = {env_key}")
print(f"api_key = {api_key}")
print("\n")

# pinecone client and embeddings
pineCone = PineconeClient(
    api_key=api_key
)

# register the API key at cohere.com (RAG site)
embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key="y8rfGb9Ak9BSWnA6GcjDfoT4V0nXETZXzk5vf7TR")
print("Embeddings:")
print(embeddings)
print("\n")

# get the index from existing srcs
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

# RAG prompt
template = """Anser the question based only on the following context:
    {context}
    Question: {question}
"""
#prompt = 
