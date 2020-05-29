#!/usr/bin/env python3
"""
Usage:
    benchmark_tcr_dist.py <SEQ_FILE> <N_SEQS> <N_JOBS>
    benchmark_tcr_dist.py sequences.txt 10000 32
"""

import sys

import scirpy as ir
import scanpy as sc
import pandas as pd
import time

seq_file = sys.argv[1]
n = int(sys.argv[2])
n_jobs = int(sys.argv[3])

seq_df = pd.read_csv(seq_file)
unique_seqs = seq_df[0][:n]

tic = time.clock()
ir.pp.tcr_dist(unique_seqs, metric='alignment', cutoff=10, n_jobs=n_jobs)
toc = time.clock()
print("Time elapsed: ", tic - toc)
