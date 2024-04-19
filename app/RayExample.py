import ray

# def for changing batches to dict
def change_batch(batch):
    return batch 

# tensor ray data
ds = ray.data.range_tensor(1000, shape=(125_000, ), parallelism=10)
new_ds = ds.map_batches(
    change_batch,
    batch_size=1000)
summary = new_ds.materialize()

print("Dataset:")
print(ds)
print("\n")

all_ds = ds.take_all()
all_sum = summary.take_all()

print(f"All ds type = {type(all_ds)}")
print(all_ds)
print("\n")

print(f"All sum type = {type(all_sum)}")
print(all_sum)
print("\n")
