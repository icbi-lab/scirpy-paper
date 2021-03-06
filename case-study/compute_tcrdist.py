#!/usr/bin/env python
#$ -cwd
#$ -S /home/sturm/.conda/envs/scirpy/bin/python
#$ -V
#$ -R y
#$ -pe smp 35
#$ -q all.q@apollo*
import sys
sys.path.append("/home/sturm/projects/2020/scirpy")
import scirpy as ir
import scanpy as sc

# print(f"Input Path: {sys.argv[1]}")
# print(f"Output Path: {sys.argv[2]}")

adata = ir.read_h5ad("adata_in.h5ad")
sc.settings.verbosity=4
# ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="primary_only", cutoff=10, key_added="ct_al_10")
# ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="primary_only", cutoff=15, key_added="ct_al_15")
# ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="primary_only", cutoff=20, key_added="ct_al_20")
# ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="all", cutoff=10, key_added="ct_al_10")
ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="all", cutoff=15, key_added="tcr_neighbors_al15", n_jobs=35)
# ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="all", cutoff=17, key_added="ct_al_20")
adata.write_h5ad("adata_alignment.h5ad")
