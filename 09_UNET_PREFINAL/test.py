
#import os
#import socket
#
#import torch
#import torch.multiprocessing as mp
#import torch.nn as nn
#from torch import distributed as dist
#from torch.nn.parallel import DistributedDataParallel
#from tqdm import tqdm
#
#
#def find_free_port():
#    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    sock.bind(("", 0))
#    port = sock.getsockname()[1]
#    sock.close()
#    return port
#
#
#def main():
#    num_available_gpus = torch.cuda.device_count()
#    assert num_available_gpus > 1, "Use more than 1 GPU for DDP"
#    world_size = num_available_gpus
#    os.environ['MASTER_ADDR'] = str("localhost")
#    os.environ['MASTER_PORT'] = str(find_free_port())
#
#    print("Distributed Test Code")
#    print(f"GPU name: {torch.cuda.get_device_name()}")
#    print(f"CUDA capability: {torch.cuda.get_device_capability()}")
#    print(f"Num GPUs: {num_available_gpus}")
#
#    mp.spawn(main_process, nprocs=num_available_gpus, args=(num_available_gpus,))
#
#
#def main_process(gpu, world_size):
#    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=gpu)
#    torch.cuda.set_device(gpu)
#    model = nn.Linear(10, 10).cuda(gpu)
#    model = DistributedDataParallel(model, device_ids=[gpu])
#
#    criterion = nn.L1Loss()
#
#    for epoch in range(10):
#        print(epoch)
#        for iteration in tqdm(range(1000), desc=f"Ep: {epoch}"):
#            print(epoch)
#            for p in model.parameters():
#                p.grad = None
#            x = torch.randn(2, 10).cuda(gpu)
#            gt = torch.randn(2, 10).cuda(gpu)
#            out = model(x)
#            loss = criterion(out, gt)
#            loss.backward()
#
#
#if __name__ == '__main__':
#    main()




