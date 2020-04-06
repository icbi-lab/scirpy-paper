---
jupyter:
  jupytext:
    formats: md,ipynb
    notebook_metadata_filter: -kernelspec
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
---

# Analysis of 140k T cells from cancer
~100k of which have TCR

<!-- #raw raw_mimetype="text/restructuredtext" -->
In this tutorial, we re-analize single-cell TCR/RNA-seq data from Wu et al (:cite:`Wu2020`)
generated on the 10x Genomics platform. The original dataset consists of >140k T cells
from 14 treatment-naive patients across four different types of cancer.
<!-- #endraw -->

### Todo
 * compare clonotype definition
 * compare alignment vs. identity
 * fraction of expanded clonotype
     - across patients
     - across clusters
     - across sources
 * identify dual-expanded clonotypes
     - abundance across patients/tumor types
     - abundance across clusters
 * identify blood-expanded clonotypes

```python
import os
n_cores = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = n_cores
os.environ["OMP_NUM_THREADS"] = n_cores
os.environ["MKL_NUM_THREADS"] = n_cores
os.environ["OMP_NUM_cpus"] = n_cores
os.environ["MKL_NUM_cpus"] = n_cores
os.environ["OPENBLAS_NUM_cpus"] = n_cores
```

```python
%load_ext autoreload
%autoreload 2
import sys

sys.path.append("../../scirpy/")
import scirpy as ir
import pandas as pd
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
import matplotlib
```

```python
def fast_subset(adata, mask):
    """Subsetting adata with columns with many categories takes forever. 
    The reason is that `remove_unused_categories` from pandas is super slow. 
    Need to report that to the pandas devs at some point or work around it
    in AnnData. 
    
    In the meanwhile, subset adata by copying it, subsetting the `obs` dataframe
    individually and re-adding it to the copy.  
    """
    adata2 = adata.copy()
    adata2.obs = pd.DataFrame(adata.obs.index)
    adata2 = adata2[mask, :].copy()
    adata2.obs = adata.obs.loc[mask, :]
    return adata2
```

```python
sc.logging.print_versions()
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
The dataset ships with the `scirpy` package. We can conveniently load it from the `dataset` module:
<!-- #endraw -->

```python
adata = ir.datasets.wu2020()
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
`adata` is a regular :class:`~anndata.AnnData` object:
<!-- #endraw -->

```python
adata.shape
```

We only keep the cells with TCR. ~96k cells remain. 

```python
adata = adata[adata.obs["has_tcr"] == "True", :]
adata = adata[~(adata.obs["cluster_orig"] == "nan"), :]
```

```python
adata.shape
```

## Preprocess Transcriptomics data

Transcriptomics data needs to be filtered and preprocessed as with any other single-cell dataset.
We recommend following the [scanpy tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
and the best practice paper by [Luecken et al.](https://www.embopress.org/doi/10.15252/msb.20188746). 
For the _Wu2020_ dataset, the authors already provide clusters and UMAP coordinates.
Instead of performing clustering and cluster annotation ourselves, we will just use
provided data.

```python
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_cells(adata, min_genes=100)
```

```python
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1000)
sc.pp.log1p(adata)
```

```python
adata.obsm["X_umap"] = adata.obsm["X_umap_orig"]
```

```python
mapping = {
    "3.1-MT": "other",
    "4.1-Trm": "CD4_Trm",
    "4.2-RPL32": "CD4_RPL32",
    "4.3-TCF7": "CD4_TCF7",
    "4.4-FOS": "CD4_FOSS",
    "4.5-IL6ST": "CD4_IL6ST",
    "4.6a-Treg": "CD4_Treg",
    "4.6b-Treg": "CD4_Treg",
    "8.1-Teff": "CD8_Teff",
    "8.2-Tem": "CD8_Tem",
    "8.3a-Trm": "CD8_Trm",
    "8.3b-Trm": "CD8_Trm",
    "8.3c-Trm": "CD8_Trm",
    "8.4-Chrom": "other",
    "8.5-Mitosis": "other",
    "8.6-KLRB1": "other",
    "nan": "nan"
}
adata.obs["cluster"] = [mapping[x] for x in adata.obs["cluster_orig"]]
```

Let's inspect the UMAP plots. The first three panels show the UMAP plot colored by sample, patient and cluster.
We don't observe any clustering of samples or patients that could hint at batch effects.

The lower three panels show the UMAP colored by the T cell markers _CD8_, _CD4_, and _FOXP3_.
We can confirm that the markers correspond to their respective cluster labels.

```python
sc.pl.umap(adata, color=["sample", "patient", "cluster", "CD8A", "CD4", "FOXP3"], ncols=2, wspace=.5)
```

## TCR Quality Control

<!-- #raw raw_mimetype="text/restructuredtext" -->
While most of T cell receptors have exactly one pair of α and β chains, up to one third of 
T cells can have *dual TCRs*, i.e. two pairs of receptors originating from different alleles (:cite:`Schuldt2019`).

Using the :func:`scirpy.tl.chain_pairing` function, we can add a summary
about the T cell receptor compositions to `adata.obs`. We can visualize it using :func:`scirpy.pl.group_abundance`.

.. note:: **chain pairing**

    - *Orphan chain* refers to cells that have either a single alpha or beta receptor chain.
    - *Extra chain* refers to cells that have a full alpha/beta receptor pair, and an additional chain.
    - *Multichain* refers to cells with more than two receptor pairs detected. These cells are likely doublets.
<!-- #endraw -->

```python
%%time
ir.tl.chain_pairing(adata)
```

```python
ir.pl.group_abundance(
    adata, groupby="chain_pairing", target_col="source",
)
```

Indeed, in this dataset, ~7% of cells have more than a one pair of productive T-cell receptors:

```python
print("Fraction of cells with more than one pair of TCRs: {:.2f}".format(
    np.sum(adata.obs["chain_pairing"].isin(["Extra beta", "Extra alpha", "Two full chains"])) / adata.n_obs
))
```

Next, we visualize the _Multichain_ cells on the UMAP plot and exclude them from downstream analysis:

```python
sc.pl.umap(adata, color="chain_pairing", groups=["Extra beta", "Extra alpha", "Two full chains"], size=[10 if x in ["Extra beta", "Extra alpha", "Two full chains"] else 3 for x in adata.obs["chain_pairing"]])
```

```python
sc.pl.umap(adata, color="chain_pairing", groups="Multichain", size=[30 if x == "Multichain" else 3 for x in adata.obs["chain_pairing"]])
```

```python
adata.shape
```

```python
adata = adata[adata.obs["chain_pairing"] != "Multichain", :].copy()
```

```python
adata.shape
```

## Define clonotypes

```python
sc.settings.verbosity = 4
```

The authors of the dataset define the clonotypes on the nucleotide sequences and require all sequences of both receptor arms (and multiple chains in case of dual TCRs) to match. 

```python
%%time
ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="all", cutoff=0, n_jobs=42, sequence="nt", key_added="tcr_neighbors_nt")
```

Commonly, clonotypes are defined based on their Amino acid sequence instead, because they recognize the same epitope. 

```python
%%time
ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="all", cutoff=0, n_jobs=42)
```

With `scirpy`, it is possible to to one step further and summarize cells into the same clonotype that likely recognize the 
same epitope. This can be done by leveraging levenshtein or alignment distances. Here, we compute the alignment distance
with a cutoff of 15, which is equivalent of three As mutating into R. 

```python
adata.write_h5ad("./adata_in.h5ad")
```

```python
adata = sc.read_h5ad("./adata_alignment3.h5ad")
```

```python
%%time
ir.tl.define_clonotypes(adata, neighbors_key="tcr_neighbors_nt", key_added="clonotype_nt")
```

```python
%%time
ir.tl.define_clonotypes(adata, neighbors_key="tcr_neighbors", key_added="clonotype")
```

```python
%%time
ir.tl.define_clonotypes(adata, neighbors_key="ct_al_15", key_added="clonotype_alignment")
```

```python
%%time
ir.tl.define_clonotypes(adata, neighbors_key="ct_al_10", key_added="clonotype_alignment_10")
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
To visualize the network we first call :func:`scirpy.tl.clonotype_network` to compute the layout.
We can then visualize it using :func:`scirpy.pl.clonotype_network`. We recommend setting the
`min_size` parameter to `>=2`, to prevent the singleton clonotypes from cluttering the network.
<!-- #endraw -->

```python
%%time
ir.tl.clonotype_network(adata, min_size=50, layout="components")
```

```python
ir.pl.clonotype_network(adata, color=["clonotype", "clonotype_nt", "clonotype_orig", "patient"], edges=False, size=50, ncols=2, legend_loc=["on data", "none", "none", "right margin"])
```

## Clonotype consistency

### clonotype_nt vs clonotype_orig

```python
# the clonotypes in clonotype_orig that contain multiple clonotype_nt
nt_in_orig = adata.obs.groupby(["clonotype_orig", "clonotype_nt"]).size().reset_index().groupby("clonotype_orig").size().reset_index()
nt_in_orig[nt_in_orig[0] > 1]
```

-> None! 

```python
# the clonotypes in clonotype_nt that contain multiple clonotype_orig
orig_in_nt = adata.obs.groupby(["clonotype_nt", "clonotype_orig"]).size().reset_index().groupby("clonotype_nt").size().reset_index()
with pd.option_context('display.max_rows', 10):
    display(orig_in_nt[orig_in_nt[0] > 1])
```

A few. Let's investigate that further. 

```python
with pd.option_context('display.max_rows', 10):
    display(adata.obs.loc[
        adata.obs["clonotype_nt"].isin(orig_in_nt[orig_in_nt[0] > 1]["clonotype_nt"]),
        ["clonotype_nt", "clonotype_orig", "patient", "TRA_1_cdr3_nt", "TRA_2_cdr3_nt", "TRB_1_cdr3_nt", "TRB_2_cdr3_nt"]
    ].sort_values(["clonotype_nt", "clonotype_orig"]))
```

All cells of the same `clonotype_nt` have the same nucleotide sequences (apart from swapping between `TRA_1` and `TRA_2` or `TRB_1` and `TRB_2`, respectively). Our method appears to work as expected. 
Most of the inconsistencies arise from the fact that the clonotypes have the same sequences, but originate from different patients. Apparently, the 
authors did not allow clonotypes from different patients (which makes sense when approaching clonotypes from a genomic point of view, not from an epitope recognition point of view). 

When checking the number of clonotypes_orig per clonotype_nt and patient, we again have no results. 
Overall the data is highly consistent. 

```python
# the clonotypes in clonotype_nt that contain multiple clonotype_orig
orig_in_nt = adata.obs.groupby(["clonotype_nt", "patient", "clonotype_orig"]).size().reset_index().groupby(["clonotype_nt", "patient"]).size().reset_index()
with pd.option_context('display.max_rows', 8):
    display(orig_in_nt[orig_in_nt[0] > 1])
```

```python
adata.obs.groupby(["clonotype_nt", "patient"]).size().reset_index().shape
```

```python
adata.obs["clonotype_orig"].unique().size
```

## Identity vs. Alignment

```python
adata_sub = adata[~adata.obs["chain_pairing"].str.startswith("Orphan"), :]
```

```python
%%time
ir.tl.clonotype_network(adata_sub, min_size=50, layout="components", neighbors_key="ct_al_15", key_clonotype_size="clonotype_alignment_size")
```

```python
ir.pl.clonotype_network(adata_sub, color=["clonotype_alignment", "clonotype", "patient"], edges=False, size=50, ncols=2, legend_loc=["on data", "none", "none", "right margin"], legend_fontoutline=2)
```

CT`1626` seems particularly interesting

```python
!pip install weblogo
```

```python
LogoData.from_seqs()
```

```python
from weblogo.seq import SeqList, protein_alphabet, unambiguous_protein_alphabet
from weblogo import png_formatter, png_print_formatter
from weblogo import LogoData, LogoOptions, LogoFormat
from IPython.display import Image, display
```

```python
def weblogo(seqs):
    logodata = LogoData.from_seqs(SeqList(seqs, alphabet=unambiguous_protein_alphabet))
    logooptions = LogoOptions()
    logooptions.title = "A Logo Title"
    logoformat = LogoFormat(logodata, logooptions)
    display(Image(png_formatter(logodata, logoformat)))
```

```python
weblogo(adata.obs.loc[adata.obs["clonotype_alignment"] == "1626", ["TRA_1_cdr3"]].values)
```

```python
weblogo(adata.obs.loc[adata.obs["clonotype_alignment"] == "1626", ["TRB_1_cdr3"]].values)
```

```python
weblogo(adata.obs.loc[adata.obs["clonotype_alignment"] == "5304", ["TRA_1_cdr3"]].values)
```

```python
weblogo(adata.obs.loc[adata.obs["clonotype_alignment"] == "5304", ["TRB_1_cdr3"]].values)
```

## 1261 appears to occur across two lung cancer patients

```python
weblogo(adata.obs.loc[adata.obs["clonotype_alignment"] == "1261", ["TRB_1_cdr3"]].values)
```

```python
weblogo(adata.obs.loc[adata.obs["clonotype_alignment"] == "1261", ["TRA_1_cdr3"]].values)
```

In [vdjdb](https://vdjdb.cdr3.net/search), we find a match for `CAV[STR][LG]QAGTALIF` as TRA sequence for Cytomegalievirus (CMV) and the closest match (2 substitutions) for TRB is CMV as well. 


## Clonal expansion
 * fraction of expanded clonotype
     - across patients
     - across clusters
     - across sources

```python
ir.pl.clonal_expansion(adata, groupby="patient", summarize_by="clonotype", show_nonexpanded=False)
```

```python
fraction_expanded = ir.tl.summarize_clonal_expansion(adata, groupby="patient", summarize_by="clonotype", normalize=True).drop("1", axis="columns").sum(axis=1)
```

```python
min(fraction_expanded), max(fraction_expanded)
```

### across clusters

```python
ir.pl.clonal_expansion(adata, groupby="cluster_orig", summarize_by="cell", show_nonexpanded=False, fig_kws={"dpi": 120})
```

## Dual- and blood expanded clonotypes
* identify dual-expanded clonotypes
     - abundance across patients/tumor types
     - abundance across clusters
 * identify blood-expanded clonotypes

```python
clonotype_membership = {ct: list() for ct in adata.obs["clonotype"]}
```

```python
for clonotype, source in zip(adata.obs["clonotype"], adata.obs["source"]):
    clonotype_membership[clonotype].append(source)
```

```python
clonotype_membership = {ct: set(sources) for ct, sources in clonotype_membership.items()}
```

```python
categories = []
for clonotype, clonotype_size, source in zip(adata.obs["clonotype"], adata.obs["clonotype_size"], adata.obs["source"]):
    if clonotype_size == 1:
        if source == "Blood":
            categories.append("blood singleton")
        elif source == "NAT":
            categories.append("NAT singleton")
        elif source == "Tumor":
            categories.append("Tumor singleton")
    elif clonotype_size >1:
        membership = clonotype_membership[clonotype]
        if "Tumor" in membership and "NAT" in membership:
            categories.append("Dual expanded")
        elif "Tumor"in membership:
            categories.append("Tumor multiplet")
        elif "NAT" in membership:
            categories.append("NAT multiplet")
        elif "Blood" in membership:
            categories.append("Blood multiplet")
            
assert len(categories) == adata.n_obs       
```

```python
adata.obs["category"] = categories
```

```python
ir.pl.clonal_expansion(adata, groupby="category")
```

## Clonotype analysis

### Clonal expansion

Let's visualize the number of expanded clonotypes (i.e. clonotypes consisting
of more than one cell) by cell-type. The first option is to add a column with the *clonal expansion*
to `adata.obs` and plot it on the UMAP plot. 

```python
ir.pl.embedding(adata, basis="umap", color=["cluster_orig", "clonotype_size"], norm=matplotlib.colors.LogNorm(), color_map="inferno", legend_loc=["on data", "none"])
```

The second option is to show the number of cells belonging to an expanded clonotype per category
in a stacked bar plot: 

```python
ir.pl.clonal_expansion(adata, groupby="patient", fraction=True, expanded_in="patient")
```

```python
ir.pl.clonal_expansion(adata, groupby="cell_type_coarse")
```

```python
ir.pl.clonal_expansion(adata, groupby="ct_source", expanded_in="source")
```

```python
ir.pl.clonal_expansion(adata, groupby="patient", fraction=True, expanded_in="patient")
```

```python
ir.pl.clonal_expansion(adata, groupby="ct_source", clip_at=4, fraction=False)
```

The same plot, normalized to cluster size: 

```python
ir.pl.clonal_expansion(adata, "cluster_orig", fig_kws={"dpi": 120})
```

```python
def fast_subset(adata, mask):
    adata2 = adata.copy()
    adata2.obs = pd.DataFrame(adata.obs.index)
    adata2 = adata2[mask, :].copy()
    adata2.obs = adata.obs.loc[mask, :]
    return adata2
```

```python
adata_tumor = fast_subset(adata, (adata.obs["source"] == "Tumor").values)
adata_nat = fast_subset(adata, (adata.obs["source"] == "NAT").values)
adata_blood = fast_subset(adata, (adata.obs["source"] == "Blood").values)
```

```python
ir.pl.clonal_expansion(adata_tumor, "cluster_orig", fig_kws={"dpi": 120}, expanded_in="cluster_orig")
```

```python
ir.pl.clonal_expansion(adata_nat, "cluster_orig", fig_kws={"dpi": 120}, expanded_in="cluster_orig")
```

```python
ir.pl.clonal_expansion(adata_blood, "cluster_orig", fig_kws={"dpi": 120}, expanded_in="cluster_orig")
```

Expectedly, the CD8+ effector T cells have the largest fraction of expanded clonotypes. 

Consistent with this observation, they have the lowest alpha diversity of clonotypes: 

```python
ax = ir.pl.alpha_diversity(adata, groupby="cluster")
```

### Clonotype abundance

<!-- #raw raw_mimetype="text/restructuredtext" -->
The function :func:`scirpy.pl.group_abundance` allows us to create bar charts for
arbitrary categorial from `obs`. Here, we use it to show the distribution of the 
ten largest clonotypes across the cell-type clusters.
<!-- #endraw -->

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="source", max_cols=10, fraction="source"
)
```

When cell-types are considered, it might be benefitial to normalize the counts
to the sample size: 

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="patient", max_cols=10, fraction="patient"
)
```

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="cluster_orig", max_cols=10, fraction="cluster_orig"
)
```

Coloring the bars by patient gives us information about public and private clonotypes: 
While most clonotypes are private, i.e. specific to a certain tissue, 
some of them are public, i.e. they are shared across different tissues. 

```python
ax = ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="sample", max_cols=10
)
ax.legend(loc=(1.1, 0.01), ncol=4, fontsize="x-small") 
```

However, none of them is shared across patients.  
This is consistent with the observation we made earlier on the clonotype network. 

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="patient", max_cols=10
)
```

## Gene usage

<!-- #raw raw_mimetype="text/restructuredtext" -->
:func:`scirpy.tl.group_abundance` can also give us some information on VDJ usage. 
We can choose any of the `{TRA,TRB}_{1,2]_{v,d,j,c}_gene` columns to make a stacked bar plot. 
We use `max_col` to limit the plot to the 10 most abundant V-genes. 
<!-- #endraw -->

```python
ir.pl.group_abundance(
    adata,
    groupby="TRB_1_v_gene",
    target_col="cluster",
    fraction=True,
    max_cols=10
)
```

We can pre-select groups by filtering `adata`:

```python
ir.pl.group_abundance(
    adata[adata.obs["TRB_1_v_gene"].isin(
        ["TRBV20-1", "TRBV7-2", "TRBV28", "TRBV5-1", "TRBV7-9"]
    ),:],
    groupby="cluster",
    target_col="TRB_1_v_gene",
    fraction=True,
)
```

<!-- #raw raw_mimetype="text/restructuredtext" -->
The exact combinations of VDJ genes can be visualized as a Sankey-plot using :func:`scirpy.pl.vdj_usage`. 
<!-- #endraw -->

```python
ir.pl.vdj_usage(adata, full_combination=False, top_n=30)
```

### Spectratype plots

<!-- #raw raw_mimetype="text/restructuredtext" -->
:func:`~scirpy.pl.spectratype` plots give us information about the length distribution of CDR3 regions. 
<!-- #endraw -->

```python
ir.pl.spectratype(adata, target_col="source", fig_kws={"dpi": 120}, fraction="source")
```

The same as line chart, normalized to cluster size: 

```python
ir.pl.spectratype(adata, target_col="cluster", fraction="cluster", viztype="line")
```

Again, to pre-select specific genes, we can simply filter the `adata` object before plotting. 

```python
ir.pl.spectratype(
    adata[adata.obs["TRB_1_v_gene"].isin(["TRBV20-1", "TRBV7-2", "TRBV28", "TRBV5-1", "TRBV7-9"]),:], 
    groupby="TRB_1_cdr3",
    target_col="TRB_1_v_gene",
    fraction="sample",
    fig_kws={'dpi': 150}
)
```

```python

```
