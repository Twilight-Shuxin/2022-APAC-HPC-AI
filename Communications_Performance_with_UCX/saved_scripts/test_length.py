import argparse
import asyncio
import cProfile
import gc
import io
import os
import pickle
import pstats
from time import monotonic as clock
import time
from typing import DefaultDict


import cupy
import numpy as np
import rmm

import ucp
import cudf

i_chunk = 0
num_chunks = 4
chunk_type = "left"
seed = num_chunks * int(chunk_type == "left") + i_chunk
cupy.random.seed(seed)

rmm.reinitialize(pool_allocator=True, initial_pool_size=None)

    # Make cupy use RMM
cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

local_sizes = ["5", "10", "60", "120", "180", "250", "350"]
for lsize in local_sizes:
    local_size = int(lsize) * 100000
    start = local_size * i_chunk
    stop = start + local_size

    df = cudf.DataFrame(
        {
            "key": cupy.random.randint(0, 100, size=local_size*num_chunks, dtype="int64")[start:stop],
            "payload": cupy.arange(local_size, dtype="int64"),
        }
    )
    left_bins = df.partition_by_hash(["key"], num_chunks)
    length = []
    header_length = []
    cnt = []
    for bin in left_bins:
        header, frames = bin.serialize()
        header["frame_ifaces"] = [f.__cuda_array_interface__ for f in frames]
        flen = 0
        cnt_i = 0
        for iface in header["frame_ifaces"]:
            cnt_i += 1
            x = 1
            for t in iface["shape"]:
                x *= int(t)
            length.append(x)
        header = pickle.dumps(header)
        header_nbytes = np.array([len(header)], dtype=np.uint64)
        header_length.append(header_nbytes)
        cnt.append(cnt_i)
    print("---------------------", local_size, "-----------------")
    print("header_sizes: ", header_length)
    print("cnts: ", cnt)
    print("====>lengths: ")
    for len_i in length:
        print(len_i)

#150012984