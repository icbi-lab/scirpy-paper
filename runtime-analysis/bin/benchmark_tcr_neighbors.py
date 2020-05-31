#!/usr/bin/env python3
"""
Usage:
    benchmark_tcr_dist.py <adata_file>
    benchmark_tcr_dist.py adata.h5ad
"""

# +
import sys

import scirpy as ir
import scanpy as sc
import pandas as pd
import timeit
# -

adata_file = sys.argv[1]

# +
# adata_file = "../results/03_simulate_adata/adata.h5ad"

# +
adata = sc.read_h5ad(adata_file)

timeit.timeit(
    lambda: ir.pp.tcr_neighbors(
        adata,
        metric="identity",
        receptor_arms="all",
        dual_tcr="primary_only",
        cutoff=0,
        sequence="nt",
    ),
    number=1,
)
# -


