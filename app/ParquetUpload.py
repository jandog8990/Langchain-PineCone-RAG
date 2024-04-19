from datetime import datetime
from datasets import load_dataset
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
import json
import ray
import requests
import pickle
import pandas as pd
import numpy as np
import os

# initialize vars for PineCone
pkl_data = 'pkl_data' 
index_name = "cohere-wikipedia" 
cpu_count = os.cpu_count()
url = "https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/default/train" 

# pkl datasets for later use
def pickle_records(batch, pkl_file):
    with open(pkl_data+pkl_file, 'wb') as f:
        pickle.dump(batch, f)

# Parquet loading datasets for wiki
def load_wiki_data():
    start = datetime.now()
    response = requests.get(url)
    input_files = json.loads(response.content)
    columns = ['id', 'title', 'text', 'url', 'emb']
    ds = ray.data.read_parquet(input_files, columns=columns, parallelism=cpu_count)
    end = datetime.now() - start
    print(f"Total time = {end}")
    print("Dataset:")
    print(ds.schema())
    print("\n")
    return ds

# create the wikipedia vector db index
def create_wiki_index():
    pc = PineconeGRPC()

    # ensure index DNE
    indexes = pc.list_indexes().indexes
    names = [_['name'] for _ in indexes]
    print("Names:")
    print(names)
    print("\n")

    # insert the wiki index if it DNE
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-west-2'))

def upload_batch(batch):
    client = PineconeGRPC()
    index = client.Index(index_name)

    #return index.upsert(vectors=zip(batch['_id'], batch['vector'], batch['metadata']))
    return index.upsert(vectors=batch)

def process_data(batch):
    print(f"Process data batch len = {len(batch)}") 
    data_records = batch.to_dict(orient='records')

    # data records upsert
    pc_records = [] 
    for record in data_records:
        newDict = {}
        newDict['id'] = str(record['id'])
        newDict['values'] = record['emb']
        newDict['metadata'] = {}
        newDict['metadata']['url'] = record['url']
        newDict['metadata']['title'] = record['title']
        newDict['metadata']['text'] = record['text']
        pc_records.append(newDict) 
   
    # upload the batch using PC index
    total_vectors = 0
    num_failures = 0
    try: 
        result = upload_batch(pc_records)
        print(f"result upserted = {result.upserted_count}") 
        total_vectors += result.upserted_count
    except Exception as e:
        print(f"Error: {e}") 
        num_failures += len(pc_records)

    # store these to the PC db
    return {'upserted': np.array([total_vectors]), 'errors': np.array([num_failures])} 

# load the subset pickle records
def run_dataset_upload(ray_ds):
    """ 
    with open(pkl_data+'/wiki_full_dataset.pkl', 'rb') as f:
        wiki_records = pickle.load(f)
    wiki_df = pd.DataFrame.from_dict(wiki_records)
    ray_ds = ray.data.from_pandas(wiki_df)
    """ 

    # map using batches
    print("Run dataset upload:") 
    new_ds = ray_ds.map_batches(
        process_data,
        batch_size=500, 
        batch_format="pandas")

    # mapping the wiki records
    print("Dataset -> execute materialize()")
    summary = new_ds.materialize()
    print(f"Summary output type = {type(summary)}")
    print(summary)
    print("\n")

# create the index if DNE
create_wiki_index()

# load and store the full wiki dataset
ds = load_wiki_data()
print("DS Schema:")
print(ds.schema())
print("\n")

# upload the DS to PC
run_dataset_upload(ds)
