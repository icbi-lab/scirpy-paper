---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0.rc1
  kernelspec:
    display_name: Python [conda env:sctcrpy2]
    language: python
    name: conda-env-sctcrpy2-py
---

```python
%load_ext autoreload
%autoreload 2
import sys

sys.path.insert(0, "../../../scirpy/")

import scirpy as ir
import scanpy as sc
import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
```

# Load and prepare data

```python
adata = ir.datasets.wu2020()
```

```python
adata = adata[adata.obs["has_tcr"] == "True", :]
```

#### Call clonotypes based on sequence identity

```python
ir.pp.tcr_neighbors(
    adata, metric="identity", receptor_arms="all", dual_tcr="primary_only", cutoff=0
)
```

```python
ir.tl.define_clonotypes(adata)
```

# Benchmarking scirpy clonotype networks

Assumptions
 * We only consider the case of 'primary only' CDR3 sequences, i.e. exactely one pair of alpha/beta chains per cell. 
 * We simulate unique CDR3 sequences by randomly mutating CDR3 amino acid sequences from Wu et al 2020 (~100k cells with TCRs).
   The resulting sequences might not have the same amino acid distribution as genuine CDR3 sequences, but they have 
   the same length distribution, which is the relevant factor for tcr_dist. 
   
### Length distribution

```python
ir.pl.spectratype(adata, cdr3_col="TRA_1_cdr3", color="has_tcr", figsize=(6, 3))
```

```python
ir.pl.spectratype(adata, cdr3_col="TRB_1_cdr3", color="has_tcr", figsize=(6, 3))
```

## 1. Benchmark alignment distance
Computing the alignment distance is relatively expensive and quadratic in the number of 
unique CDR3 amino acid sequences. Computing the alignment for each pair of 
CDR3 sequences in quadratic in the length of the sequences. 
By making sure that the simulated sequences follow the same length distribution as
genuine sequences, the benchmark should be representative. 


### Generate a pool of ~1M unique CDR3 sequences 

```python
aas = {
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "E": "GLU",
    "Q": "GLN",
    "D": "ASP",
    "N": "ASN",
    "H": "HIS",
    "W": "TRP",
    "F": "PHE",
    "Y": "TYR",
    "R": "ARG",
    "K": "LYS",
    "S": "SER",
    "T": "THR",
    "M": "MET",
    "A": "ALA",
    "G": "GLY",
    "P": "PRO",
    "C": "CYS",
}
alphabet = np.array(list(aas.keys()))
```

```python
def mutate(seq):
    """Randomly mutate 1-5 amino acids"""
    try:
        seq = np.array(list(seq))
        n_mut = np.random.randint(0, min(seq.size-1, 6))
        mut_inds = np.unique(np.random.randint(0, seq.size, size=n_mut)), 
        replace_aas = np.random.choice(alphabet, size=len(mut_inds), replace=True)
        seq[mut_inds] = replace_aas
    except:
        print(seq, mut_inds, replace_aas)
        assert False
    return "".join(seq)
```

```python
random.seed(42)
```

```python
tras = []
for tra in adata.obs["TRA_1_cdr3"].unique():
    for i in range(40):
        tras.append(mutate(tra))    
```

```python
tra_uq = np.unique(np.array(tras))
```

```python
tra_uq.shape
```

```python
random.seed(42)
```

```python
trbs = []
for trb in adata.obs["TRB_1_cdr3"].unique():
    for i in range(40):
        trbs.append(mutate(trb))    
```

```python
trb_uq = np.unique(np.array(trbs))
trb_uq.shape
```

We store a list of 1M unique sequences and subsample it in the runner script. 

```python
pd.DataFrame().assign(tra=tra_uq[:1000000]).to_csv("tmp/1M_cdr3.txt")
```

## Generate objects for tcr_neighbors

 * The performance of `scirpy.pp.neighbors` is limited by the number of non-negative entries of the cell x cell 
   connectivity matrix. 
 * This number depends on the number of total cells, the fraction of expanded clonotypes and the clonotype sizes. 

```python
n = [5000, 10000, 20000, 40000, 60000, 80000, adata.shape[0]]
```

```python
top_10_ct_size = []
frac_expanded = []
for i in n:
    tmp_adata = adata.copy()
    sc.pp.subsample(tmp_adata, n_obs=i)
    ir.tl.define_clonotypes(tmp_adata)
    frac_expanded.append(np.sum(tmp_adata.obs["clonotype_size"] >= 2)/ adata.shape[0])
    top_10_ct_size.append(tmp_adata.obs["clonotype_size"].sort_values(ascending=False).unique()[:10])
```

```python
plt.plot(n, [x for x in top_10_ct_size])
```

```python
plt.plot(n, frac_expanded)
```

```python
sc.pp.subsample(tmp_adata, n_obs=5000)
```

```python
tmp_adata.obs["TRA_1_cdr3"].unique().size
```

```python

```
