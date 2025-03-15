import torch

def setup(local_rank, world_size):
    """
    bind the process to a GPU and initialize the process group
    But we do not need the communication among process, dist.init_process_group is unnecessary
    """
    # dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    # we can set process number more than GPU number, but we need to bind the process to a GPU
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    print(f"Process {local_rank} / {world_size} is using GPU {local_rank % torch.cuda.device_count()} in its node.")