#!/usr/bin/env python3

import numpy as np
import os
import json

np.random.seed(0xCA7CAFE)

script_dir = os.path.dirname(os.path.realpath(__file__))

tilesize = 4
sizes = [512,1024,2048,4096,8192]
test_sizes=[(tilesize,tilesize),(tilesize,2*tilesize),(2*tilesize,tilesize),(tilesize*2,tilesize*2), (128,128)]



for (size_i,size_j) in test_sizes:
    a = np.random.randn(size_i, size_j).astype(np.float32)
    a_fname = os.path.join(script_dir, f"test_a_{size_i}_{size_j}.bin")
    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    x = np.random.randn(size_i, size_j).astype(np.float32)
    x_fname = os.path.join(script_dir, f"test_x_{size_i}_{size_j}.bin")
    with open(x_fname, "wb") as f:
        f.write(x.tobytes())

for size_i in sizes:
    a = np.random.randn(size_i, size_i).astype(np.float32)
    a_fname = os.path.join(script_dir, f"test_a_{size_i}_{size_i}.bin")
    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    x = np.random.randn(size_i,size_i).astype(np.float32)
    x_fname = os.path.join(script_dir, f"test_x_{size_i}_{size_i}.bin")
    with open(x_fname, "wb") as f:
        f.write(x.tobytes())