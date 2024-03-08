from datetime import datetime
from datasets import load_dataset
from pinecone.grpc import PineconeGRPC
import json
import ray
import requests
import pickle
import pandas as pd
import numpy as np

def query_vectors():
    client = PineconeGRPC()
    index_name = "cohere-wikipedia" 
    index = client.Index(index_name)
    resp = index.query(top_k=20)
    print("response:")
    print(resp)
    print("\n")
query_vectors()
