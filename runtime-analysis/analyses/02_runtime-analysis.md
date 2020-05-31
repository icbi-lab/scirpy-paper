---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python [conda env:.conda-scirpy_runtime_benchmark]
    language: python
    name: conda-env-.conda-scirpy_runtime_benchmark-py
---

```python
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scirpy as ir
from pathlib import Path
from dateutil import parser
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
```

# Benchmark tcr_dist

 * The performance of `tcr_dist`, the function computing the alignment-distances is limited by the number of unique TCR-alpha and -beta sequences
 * The function is in $O(n^2)$, where $n$ is the number of unique sequences

```python
adata = ir.datasets.wu2020()
adata = adata[adata.obs["has_tcr"] == "True", :]
```

```python
print(f"Number of unique alpha sequences: {adata.obs['TRA_1_cdr3'].unique().size}")
print(f"Fraction of unique alpha sequences: {adata.obs['TRA_1_cdr3'].unique().size / adata.shape[0]:.2f}")
print(f"Number of unique beta sequences: {adata.obs['TRB_1_cdr3'].unique().size}")
print(f"Fraction of unique beta sequences: {adata.obs['TRB_1_cdr3'].unique().size/adata.shape[0]:.2f}")
```

 * -> We could expect a dataset of ~1M cells to have between 300k and 500k unique sequences
 * In the benchmark, we tested up to 300k unique sequences
 * Each dataset was ran on 16 cores on an Intel E5-2699A v4, 2.4 GHz CPU


 * parse the results from the `/usr/bin/time -v` command. 

```python
results = Path("../results/02_tcr_dist")
perf = dict()
for f in results.glob("timing_mem_*.txt"):
    n = int(f.stem.split("_")[2])
    perf[n] = dict()
    with open(f, 'r') as fh:
        for l in fh.readlines():
            if "Elapsed (wall clock) time" in l:
                time_str = l.split(': ')[1].strip().split(".")[0]
                try:
                    perf[n]["wall_time"] = datetime.strptime(time_str, "%M:%S")
                except ValueError:
                    perf[n]["wall_time"] = datetime.strptime(time_str, "%H:%M:%S")
            if "Maximum resident set size (kbytes):" in l:
                perf[n]["max_mem"] = float(l.split(':')[1].strip())
```

```python
def get_time(dt):
    """in hours"""
    return (dt.hour * 3600 + dt.minute * 60 + dt.second) / 3600
```

```python
ns = sorted(perf.keys())
ns
```

```python
plt.plot(ns, [get_time(perf[n]["wall_time"]) for n in ns], 'o-')
plt.xlabel("number of unique sequence")
plt.ylabel("wall time (hours)")
```

```python
plt.plot(ns, [perf[n]["max_mem"] / 1024**2 for n in ns], 'o-', color="orange")
plt.xlabel("number of unique sequence")
plt.ylabel("max. memory usage (GiB)")
```

# TCR neighbors

```python
results = Path("../results/04_tcr_neighbors")
perf = dict()
for f in results.glob("timing_mem_adata_*.txt"):
    n = int(f.stem.split("_")[3])
    perf[n] = dict()
    with open(f, 'r') as fh:
        for l in fh.readlines():
            if "Elapsed (wall clock) time" in l:
                time_str = l.split(': ')[1].strip().split(".")[0]
                try:
                    perf[n]["wall_time"] = datetime.strptime(time_str, "%M:%S")
                except ValueError:
                    perf[n]["wall_time"] = datetime.strptime(time_str, "%H:%M:%S")
            if "Maximum resident set size (kbytes):" in l:
                perf[n]["max_mem"] = float(l.split(':')[1].strip())
```

```python
ns = sorted(perf.keys())
ns
```

```python
plt.plot(ns, [get_time(perf[n]["wall_time"]) for n in ns], 'o-')
plt.xlabel("number of unique sequence")
plt.ylabel("wall time (hours)")
```

```python
plt.plot(ns, [perf[n]["max_mem"] / 1024**2 for n in ns], 'o-', color="orange")
plt.xlabel("number of unique sequence")
plt.ylabel("max. memory usage (GiB)")
```

```python

```
