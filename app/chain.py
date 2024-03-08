#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# In[3]:


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

# In[4]:


pineCone = PineconeClient(
    api_key=api_key
)


# start to load the data

# In[5]:


url = "https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/default/train"
response = requests.get(url)
input_files = json.loads(response.content)
columns = ['id', 'title', 'text', 'url', 'emb']
ds = ray.data.read_parquet(input_files, columns=columns) 
print("Dataset:")
print(ds)
print("\n")


# In[6]:

"""
ds.schema()
count = 0
for row in ds.iter_rows():
    print(row)
    count = count + 1
    if count > 0:
        break
"""


# create the index for wikipedia data

# 
# <br>
# indexes = pc.list_indexes().indexes<br>
# print("Indexes:")<br>
# print(indexes)<br>
# print("\n")<br>
# 

# 
# <br>
# embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)<br>
# print("Embeddings:")<br>
# print(embeddings)<br>
# print("\n")<br>
# 

# get the index from existing srcs

# 
# <br>
# vectorstore = Pinecone.from_existing_index(index_name=pc_index,<br>
#         embedding=embeddings)<br>
# retriever = vectorstore.as_retriever()<br>
# print("Vector store:")<br>
# print(vectorstore)<br>
# print("\n")<br>
# def fetch_wiki_page(id):<br>
#     url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&pageids={id}"<br>
#     response = requests.get(url)<br>
#     data = response.json()<br>
#     page_content = list(data['query']['pages'].values())[0]['extract']<br>
#     return page_content<br>
# def fetch_url(x):<br>
#     urls = [doc.metadata['url'] for doc in x['context']]<br>
#     ids = [url.split('=')[-1] for url in urls]<br>
#     contents = [fetch_wikipedia_page(id)[:32000] for id in ids]<br>
#     return {"context": contents, "question": x["question"]}<br>
# 

# RAG prompt

# template = 
# Answer the question based only on the following context:<br>
#     {context}<br>
#     Question: {question}<br>
# <br>
# rompt = ChatPromptTemplate.from_template(template)

# RAG model<br>
# odel = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", openai_api_key=openai_api_key)

# 
# <br>
# chain = (<br>
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})<br>
#     | RunnableLambda(fetch_url)<br>
#     | prompt<br>
#     | model<br>
#     | StrOutputParser()<br>
# )<br>
# 

# In[7]:


# upsert the data
"""
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

pc = PineconeGRPC()
index_name = 'cohere-wikipedia'

# ensure index DNE
indexes = pc.list_indexes().indexes
names = [_['name'] for _ in indexes]
print("Names:")
print(names)
print("\n")


# In[8]:


# insert the new wiki index
if index_name not in names:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-west-2'))
"""


# In[9]:


import numpy as np
from retry import retry
from tqdm.auto import tqdm

# process the data into vector format
def create_pc_dataset(dataset):
    pcDataList = []
    for row in dataset.iter_rows():
        newRow = {}
        newRow['_id'] = row['id']
        newRow['metadata'] = {}
        newRow['metadata']['title'] = row['title']
        newRow['metadata']['text'] = row['text']
        newRow['vector'] = row['emb']
        pcDataList.append(newRow)
    return pcDataList
        
def upload_batches(dataset):
    batch_size=350

# insert records async
def upload(batch):
    client = PineconeGRPC()
    index = client.Index(index_name)

    # sets the returned and error vectors
    total_vectors = 0
    num_failures = 0

    # data = process_data(large_batch).to_dict(orient='records')
    data = batch.to_dict(orient='records')

    # this will retry up to 2 times, exponential wait increase from min to 4-10
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
    def send_batch(batch):
        return index.upsert(vectors=batch)

    try:
        result = send_batch(data)
        total_vectors += result.upserted_count
    except Exception as e:
        logging.exception(e)
        num_failures += len(data)
    return {'upserted': np.array([total_vectors]), 'errors': np.array([num_failures])}

class Upserter:
    def __call__(self, large_batch):
        return upload_batches(large_batch)


# In[13]:


# set the env var for ray memory error
#get_ipython().run_line_magic('env', 'RAY_memory_monitor_refresh_ms=0')


# In[15]:


# run the upload batches
from datetime import datetime
from datasets import Dataset
import pickle

#get_ipython().run_line_magic('env', 'RAY_memory_monitor_refresh_ms=0')

# create the new dataset
pcDataList = create_pc_dataset(ds)
print(f"PC Data list len = {len(pcDataList)}")
pcDataset = Dataset.from_list(pcDataList)

# create a pkl file for use later so we don't keep reloading
pkl_data = 'pkl_data'
with open(pkl_data+'/pc_dataset.pkl', 'wb') as f:
    pickle.dump(pcDataset, f)

# upload the new ds in batches

# new_ds = ds.map_batches(
#     Upserter,
#     batch_size=batch_size,
#     batch_format='pandas',
#     zero_copy_batch=True,
#     concurrency=1)

# before = datetime.now()
# summary = new_ds.materialize().sum(['upserted', 'errors'])

# summary
# duration = datetime.now() - before
# print({k: f"{v: ,}" for k,v in summary.items()})
# print(f"Duration = {duration}")


# In[ ]:





# In[ ]:




