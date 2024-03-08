from typing import Dict
import numpy as np
import ray

ds = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")
print(f"DS type = {type(ds)}")
print(ds.schema())
print(f"DS count = {ds.count()}")
ds_batch = ds.take_batch(10)
print(f"DS Batch count = {ds_batch.count()}")
print(ds_batch.schema())
print("\n")

# batch func for chunks??
def process_data(batch):
    print("Process Batch Data:") 
    print(f"Batch type = {type(batch)}") 
    print(f"Batch size = {batch.size}")
    print("Batch:")
    print(batch)
    print("\n") 
    data_records = batch.to_dict(orient='records')
    print(f"Data records type = {type(data_records)}")
    print(data_records)
    print("\n\n")

    """ 
    data_dict = batch.to_dict(orient='dict')
    recKey = list(data_records.keys())[0]
    dicKey = list(data_dict.keys())[0]
    print(f"Rec: key = {recKey}, val = {data_records[recKey]}")
    print(f"Dic: key = {dicKey}, val = {data_dict[dicKey]}")
    print("\n")
    """ 
    return {'results': np.array(data_records)} 

# try mapping
new_ds = ds_batch.map_batches(
    process_data,
    batch_format="pandas")

# run the batch mapping
print(f"New ds type = {type(new_ds)} => execute materialize()")
summary = new_ds.materialize()
print("Summary output:")
print(f"Output type = {type(summary)}")
print(summary.schema())
print("\n")

