import torch
import torch.multiprocessing as mp
print(torch.__config__.parallel_info())


num_processes = 3
model = MyModel()
model.share_memory()
processes = []
for rank in range(num_processes):
    p = mp.Process(target=train, args=(model,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
