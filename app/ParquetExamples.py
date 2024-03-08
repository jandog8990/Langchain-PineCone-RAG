from datetime import datetime
from datasets import load_dataset
from pinecone.grpc import PineconeGRPC
import json
import ray
import requests
import pickle
import pandas as pd
import numpy as np

# Examples of loading datasets 
"""
start = datetime.now()
docs = load_dataset(f"Cohere/wikipedia-22-12-en-embeddings", split="train")
end = datetime.now()

print(f"Duration = {start-end}")
print(f"Docs type = {type(docs)}")
print(f"Docs len = {len(docs)}")
"""

pkl_data = 'pkl_data' 
def pickle_records(batch):
    with open(pkl_data+'/wiki_dataset.pkl', 'wb') as f:
        pickle.dump(batch, f)

# Parquet loading datasets for wiki
"""
url = "https://huggingface.co/api/datasets/Cohere/wikipedia-22-12-en-embeddings/parquet/default/train" 
start = datetime.now()
response = requests.get(url)
input_files = json.loads(response.content)
columns = ['id', 'title', 'text', 'url', 'emb']
ds = ray.data.read_parquet(input_files, columns=columns)
end = datetime.now() - start
print(f"Total time = {end}")
print("Dataset:")
print(ds)
print("\n")

new_ds = ds.take_batch(20)
print("New Dataset:")
print(new_ds)
print("\n")
pickle_records(new_ds)
"""

def upload_batch(batch):
    client = PineconeGRPC()
    index_name = "cohere-wikipedia" 
    index = client.Index(index_name)
    
    # upsert batch to pinecone
    print(f"Upload batch len = {len(batch)}") 
    print("index batch:")
    print(batch)
    print("\n")

    #return index.upsert(vectors=zip(batch['_id'], batch['vector'], batch['metadata']))
    return index.upsert(vectors=batch)

def process_data(batch):
    print("Batch:")
    print(f"Batch type = {type(batch)}")
    print(f"Batch size = {batch.size}")
    print(batch)
    print("\n")
   
    data_records = batch.to_dict(orient='records')
    print(f"Data records type = {type(data_records)}")
    print(f"OG records len = {len(data_records)}")
    print("\n\n")

    # data records upsert
    pc_records = [] 
    for record in data_records:
        newDict = {}
        newDict['id'] = str(record['id'])
        newDict['values'] = record['emb']
        newDict['metadata'] = {}
        newDict['metadata']['title'] = record['title']
        newDict['metadata']['text'] = record['text']
        pc_records.append(newDict) 
    print(f"New data records type = {type(pc_records)}")
    print(f"New data records len = {len(pc_records)}")
    print("\n\n")
   
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

# load the pickle records
with open(pkl_data+'/wiki_dataset.pkl', 'rb') as f:
    wiki_records = pickle.load(f)
wiki_df = pd.DataFrame.from_dict(wiki_records)
ray_ds = ray.data.from_pandas(wiki_df)
print("Wiki records:")
print(f"len records = {len(wiki_records)}")
print(f"wiki type = {type(wiki_records)}")
print(f"wiki df = {type(wiki_df)}")
print("\n")
print(wiki_df)
print("\n")
print(f"Ray ds type = {type(ray_ds)}")
print(ray_ds)
print("\n")

# map using batches
new_ds = ray_ds.map_batches(
    process_data,
    batch_size=5, 
    batch_format="pandas")

# mapping the wiki records
print("Dataset -> execute materialize()")
summary = new_ds.materialize()
print(f"Summary output type = {type(summary)}")
print(summary)
print("\n")

"""
ray_list = wiki_records.items()
print(f"Ray list len = {len(ray_list)}")
print(f"ray type list = {type(ray_list)}")
print(ray_list[0])
print("\n")

ray_ds = ray.data.from_items(ray_list)
print("Ray ds:")
print(ray_ds.schema())
print("\n")
"""
