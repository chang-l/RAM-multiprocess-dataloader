#!/usr/bin/env python
import itertools
import multiprocessing as mp
import os
import time
import torch
import torch.distributed as dist

import detectron2.utils.comm as comm
from detectron2.engine import launch

from common import MemoryMonitor, create_coco, DatasetFromList, read_sample
from serialize import TorchShmSerializedList

def create_tensors():
    tensors = [torch.rand(10) for _ in range(2)]
    return tensors

def local_broadcast_process_authkey():
    local_rank = int(os.environ['LOCAL_RANK'])
    authkey = bytes(mp.current_process().authkey)
    all_keys = comm.all_gather(authkey)
    local_leader_key = all_keys[comm.get_rank() - local_rank]
    if authkey != local_leader_key:
        print("Process authkey is different from the key of local leader. This might happen when "
              "workers are launched independently.")
        print("Overwriting local authkey ...")
        mp.current_process().authkey = local_leader_key

def main():
  dist.init_process_group(backend='gloo')
  comm.create_local_process_group(dist.get_world_size())

  local_broadcast_process_authkey()
  monitor = MemoryMonitor()
  ds = DatasetFromList(TorchShmSerializedList(
      # Don't read data except for GPU worker 0! Otherwise we waste time and (maybe) RAM.
      create_tensors() if comm.get_local_rank() == 0 else []))

if __name__ == "__main__":
  # This uses "spawn" internally. To switch to forkserver, modifying
  # detectron2 source code is needed.
#  launch(main, num_gpus, dist_url="auto")
  main()
