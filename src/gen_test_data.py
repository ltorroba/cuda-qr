#!/usr/bin/env python3

import numpy as np
import os
import json

np.random.seed(0xCA7CAFE)

script_dir = os.path.dirname(os.path.realpath(__file__))


sizes = [ 32, 128,  512, 2048]
test_sizes=[(32,32),(32,64),(64,32),(64,64)]



with open(os.path.join(script_dir, "sizes.json"), "w") as f:
    json.dump(
        [
            {"size": size_i, "size_j":size_i}
            for size_i in sizes
        ],
        f,
        indent=2,
    )
with open(os.path.join(script_dir, "test_sizes.json"), "w") as f:
    json.dump(
        [
            {"size_i": size_i, "size_j":size_j}
            for (size_i,size_j) in test_sizes
        ],
        f,
        indent=2,
    )



for size_i in sizes:
    a = np.random.randn(size_i, size_i).astype(np.float32)
    a_fname = os.path.join(script_dir, f"test_a_{size_i}_{size_i}.bin")
    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    print(f"Wrote {a_fname!r}")
    c = np.transpose(np.linalg.qr(a, mode = "raw")[0])
    c_fname = os.path.join(script_dir, f"test_ref_{size_i}_{size_i}.bin")
    with open(c_fname, "wb") as f:
        f.write(c.tobytes())
    print(f"Wrote {c_fname!r}")

for (size_i,size_j) in test_sizes:
    a = np.random.randn(size_i, size_j).astype(np.float32)
    a_fname = os.path.join(script_dir, f"test_a_{size_i}_{size_j}.bin")
    with open(a_fname, "wb") as f:
        f.write(a.tobytes())
    print(f"Wrote {a_fname!r}")
    c = np.transpose(np.linalg.qr(a, mode = "raw")[0])
    c_fname = os.path.join(script_dir, f"test_ref_{size_i}_{size_j}.bin")
    with open(c_fname, "wb") as f:
        f.write(c.tobytes())
    print(f"Wrote {c_fname!r}")
