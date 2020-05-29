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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
```

```python
adata.shape
```

```python
adata.obs["TRA_1_cdr3"].unique().size
```

```python
frac_unique_alpha = adata.obs["TRA_1_cdr3"].unique().size / adata.shape[0]
```

```python
frac_unique_beta = adata.obs["TRB_1_cdr3"].unique().size / adata.shape[0]
```
