---
jupyter:
  jupytext:
    formats: md,ipynb
    notebook_metadata_filter: -kernelspec
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.2
---

# Analysis of 140k T cells from cancer

<!-- #region raw_mimetype="text/restructuredtext" -->
In this notebook, we re-analize single-cell TCR/RNA-seq data from [Wu et al. (2020)](https://www.nature.com/articles/s41586-020-2056-8)
generated on the 10x Genomics platform. The original dataset consists of >140k T cells
from 14 treatment-naive patients across four different types of cancer. Roughly 100k of the 140k cells have T-cell receptors. 
<!-- #endregion -->

## 0. Setup

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
from weblogo.seq import SeqList, unambiguous_protein_alphabet
from weblogo import png_formatter
from weblogo import LogoData, LogoOptions, LogoFormat
from IPython.display import Image, display
```

```python
sc.logging.print_versions()
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
def weblogo(seqs):
    """Draw a sequence logo from a list of amino acid sequences. """
    logodata = LogoData.from_seqs(SeqList(seqs, alphabet=unambiguous_protein_alphabet))
    logooptions = LogoOptions()
    logooptions.title = "A Logo Title"
    logoformat = LogoFormat(logodata, logooptions)
    display(Image(png_formatter(logodata, logoformat)))
```

## 1. Preparing the data

<!-- #region raw_mimetype="text/restructuredtext" -->
The dataset ships with the `scirpy` package. We can conveniently load it from the `dataset` module. 
<!-- #endregion -->

```python
adata = ir.datasets.wu2020()
```

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

### Preprocess Transcriptomics data

Transcriptomics data needs to be filtered and preprocessed as with any other single-cell dataset.
We recommend following the [scanpy tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
and the best practice paper by [Luecken et al.](https://www.embopress.org/doi/10.15252/msb.20188746). 
For the _Wu et al. (2020)_ dataset, the authors already provide clusters and UMAP coordinates.
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

<!-- #region raw_mimetype="text/restructuredtext" -->
While most of T cell receptors have exactly one pair of α and β chains, up to one third of 
T cells can have *dual TCRs*, i.e. two pairs of receptors originating from different alleles ([Schuldt et al (2019)](https://doi.org/10.4049/jimmunol.1800904).

Using the `scirpy.tl.chain_pairing` function, we can add a summary
about the T cell receptor compositions to `adata.obs`.


- *Orphan chain* refers to cells that have either a single alpha or beta receptor chain.
- *Extra chain* refers to cells that have a full alpha/beta receptor pair, and an additional chain.
- *Multichain* refers to cells with more than two receptor pairs detected. These cells are likely doublets.
<!-- #endregion -->

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
sc.pl.umap(adata,
           color="chain_pairing", 
           groups=["Extra beta", "Extra alpha", "Two full chains"],
           size=[10 if x in ["Extra beta", "Extra alpha", "Two full chains"] else 3 for x in adata.obs["chain_pairing"]])
```

```python
print("Fraction of cells with more than one pair of TCRs: {:.2f}".format(
    np.sum(adata.obs["chain_pairing"].isin(["Extra beta", "Extra alpha", "Two full chains"])) / adata.n_obs
))
```

### Excluding multichain cells
Next, we visualize the _Multichain_ cells on the UMAP plot and exclude them from downstream analysis:

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

Defining clonotypes in `scirpy` is a two-step procedure: 

 1. Computing a neighborhood graph based on CDR3 sequences
 2. Finding connected submodules in the neighborhood graph and annotating them as clonotypes
 
`scirpy` provides several metrics for creating the neighborhood graph. For instance, it is possible to choose between
using nucleotide or amino acid CDR3 sequences, or using a sequence similarity metric based on multiple 
sequence alignments instead of requiring sequences to be identical. 

```python
sc.settings.verbosity = 4
```

### identical nucleotide sequences
The authors of the dataset define the clonotypes on the nucleotide sequences and require all sequences of both receptor arms (and multiple chains in case of dual TCRs) to match. 

```python
%%time
ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="all", cutoff=0, n_jobs=42, sequence="nt", key_added="tcr_neighbors_nt")
```

```python
%%time
ir.tl.define_clonotypes(adata, neighbors_key="tcr_neighbors_nt", key_added="clonotype_nt")
```

### identical amino acid sequences
Commonly, clonotypes are defined based on their amino acid sequence instead, because they recognize the same epitope. This is the `scirpy` default. 

```python
%%time
ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="all", cutoff=0, n_jobs=42)
```

```python
%%time
ir.tl.define_clonotypes(adata, neighbors_key="tcr_neighbors", key_added="clonotype")
```

### similar amino acid sequences

With `scirpy`, it is possible to to one step further and summarize cells into the same clonotype that likely recognize the 
same epitope. This can be done by leveraging levenshtein or alignment distances. Here, we compute the alignment distance
with a cutoff of 15, which is equivalent of three `A`s mutating into `R`. 

```python
adata.write_h5ad("./adata_in.h5ad")
```

```python
# %%time
# ir.pp.tcr_neighbors(adata, receptor_arms="all", dual_tcr="all", cutoff=15, key_added="tcr_neighbors_al15")
```

```python
adata = sc.read_h5ad("./adata_alignment.h5ad")
```

```python
%%time
ir.tl.define_clonotypes(adata, neighbors_key="tcr_neighbors_al15", key_added="clonotype_alignment")
```

<!-- #region raw_mimetype="text/restructuredtext" -->
## Visualizing clonotype networks
To visualize the network we first call `scirpy.tl.clonotype_network` to compute the layout.
We can then visualize it using `scirpy.pl.clonotype_network`. 

The following plot visualizes all clonotypes with at least 50 cells. Each dot represents a cell, 
and each blob a clonotype. In the left panel each clonotype is labelled, in the right panel 
the clonotypes are colored by patient. 
<!-- #endregion -->

```python
%%time
ir.tl.clonotype_network(adata, min_size=50, layout="components")
```

```python
ir.pl.clonotype_network(adata,
                        color=["clonotype", "patient"],
                        edges=False, size=50, ncols=2, 
                        legend_fontoutline=2,
                        legend_loc=["on data", "right margin"])
```

## Clonotype consistency

Before we dive into the analysis of clonal expansion, we compare the different approaches of clonotype definition. 

### nucleotide based (scirpy vs. Wu et al.)
In this section, we compare the clonotypes assigned by `scirpy` assigned to the clonotypes assigned by the authors of the study. 
The original clonotypes are stored in the `clonotype_orig` column of `obs`. 

According to the paper, the clonotypes are defined on nucleotide sequences and require sequence identity of all available chains. 
This should be equivalent of running `scirpy.tcr_neighbors` with `dual_tcr="all"` and `receptor_arms="all"`. 

To assess if the clonotype definitions are equivalent, we first check if there are *clonotypes in `clonotype_orig` that contain multiple clonotypes according to scirpy's definition*. 

```python
# the clonotypes in clonotype_orig that contain multiple clonotype_nt
nt_in_orig = adata.obs.groupby(["clonotype_orig", "clonotype_nt"]).size().reset_index().groupby("clonotype_orig").size().reset_index()
nt_in_orig[nt_in_orig[0] > 1]
```

There are none! 

Next, we check if there are *clonotypes in `clonotype_nt` that contain multiple clonotypes according to the authors' definition.*

```python
# the clonotypes in clonotype_nt that contain multiple clonotype_orig
orig_in_nt = adata.obs.groupby(["clonotype_nt", "clonotype_orig"]).size().reset_index().groupby("clonotype_nt").size().reset_index()
with pd.option_context('display.max_rows', 10):
    display(orig_in_nt[orig_in_nt[0] > 1])
```

There are a few! 

Let's investigate that further: 

```python
with pd.option_context('display.max_rows', 10):
    display(adata.obs.loc[
        adata.obs["clonotype_nt"].isin(orig_in_nt[orig_in_nt[0] > 1]["clonotype_nt"]),
        ["clonotype_nt", "clonotype_orig", "patient", "TRA_1_cdr3_nt", "TRA_2_cdr3_nt", "TRB_1_cdr3_nt", "TRB_2_cdr3_nt"]
    ].sort_values(["clonotype_nt", "clonotype_orig"]))
```

All cells of the same `clonotype_nt` have the same nucleotide sequences (apart from swapping between `TRA_1` and `TRA_2` or `TRB_1` and `TRB_2`, respectively). Our method appears to work as expected. 
The inconsistencies seem to arise from the fact that the clonotypes have the same sequences, but originate from different patients. Apparently, the 
authors did not allow clonotypes from different patients (which makes sense when approaching clonotypes from a genomic point of view, not from an epitope recognition point of view). 

When checking the number of clonotypes_orig per clonotype_nt and patient, we eradicate the differences: 

```python
# the clonotypes in clonotype_nt that contain multiple clonotype_orig
orig_in_nt = adata.obs.groupby(["clonotype_nt", "patient", "clonotype_orig"]).size().reset_index().groupby(["clonotype_nt", "patient"]).size().reset_index()
with pd.option_context('display.max_rows', 8):
    display(orig_in_nt[orig_in_nt[0] > 1])
```

Now, also the number of clonotypes is consistent

```python
print(f"""Number of clonotypes according to Wu et al.: {adata.obs["clonotype_orig"].unique().size}""")
print(f"""Number of clonotypes according to scirpy: {adata.obs.groupby(["clonotype_nt"]).size().reset_index().shape[0]}""")
print(f"""Number of clonotypes according to scirpy, within patient: {adata.obs.groupby(["patient", "clonotype_nt"]).size().reset_index().shape[0]}""")
```

### amino-acid vs. nucleotide-based


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

```

```python

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
ir.pl.clonal_expansion(adata, groupby="patient", summarize_by="cell", show_nonexpanded=True)
```

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
ir.pl.clonal_expansion(adata, groupby="cluster_orig", summarize_by="cell", show_nonexpanded=True, fig_kws={"dpi": 120})
```

```python
ir.pl.clonal_expansion(adata, groupby="source", summarize_by="cell", show_nonexpanded=True, fig_kws={"dpi": 120})
```

## Dual- and blood expanded clonotypes
* identify dual-expanded clonotypes
     - abundance across patients/tumor types
     - abundance across clusters
 * identify blood-expanded clonotypes

```python
clonotype_size_by_source = adata.obs.groupby(["patient", "source", "clonotype"], observed=True).size().reset_index(name="clonotype_count_by_source")
adata.obs = adata.obs.reset_index().merge(clonotype_size_by_source, how="left").set_index('index')
```

```python
blood_expanded = []
for is_expanded, source in zip((adata.obs["source"] == "Blood") & (adata.obs["clonotype_count_by_source"] > 1), adata.obs["source"]):
    if source == "Blood":
        if is_expanded:
            blood_expanded.append("expanded")
        else:
            blood_expanded.append("not expanded")
    else:
        blood_expanded.append("independent")
```

```python
adata.obs["blood_expanded"] = blood_expanded
```

```python
sc.pl.umap(adata, color="blood_expanded", groups=["expanded", "not expanded"], size=[15 if x in ["expanded", "not expanded"] else 3 for x in adata.obs["blood_expanded"]])
```

```python
clonotype_membership = {ct: list() for ct in adata.obs["clonotype"]}
for clonotype, source in zip(adata.obs["clonotype"], adata.obs["source"]):
    clonotype_membership[clonotype].append(source)
clonotype_membership = {ct: set(sources) for ct, sources in clonotype_membership.items()}
```

```python
expansion_category = []
for clonotype, clonotype_size, source in zip(adata.obs["clonotype"], adata.obs["clonotype_size"], adata.obs["source"]):
    if clonotype_size == 1:
        if source == "Blood":
            expansion_category.append("Blood singleton")
        elif source == "NAT":
            expansion_category.append("NAT singleton")
        elif source == "Tumor":
            expansion_category.append("Tumor singleton")
    elif clonotype_size >1:
        membership = clonotype_membership[clonotype]
        if "Tumor" in membership and "NAT" in membership:
            expansion_category.append("Dual expanded")
        elif "Tumor"in membership:
            expansion_category.append("Tumor multiplet")
        elif "NAT" in membership:
            expansion_category.append("NAT multiplet")
        elif "Blood" in membership:
            # these are *only* expanded in blood
            expansion_category.append("Blood multiplet")
            
assert len(expansion_category) == adata.n_obs       
```

```python
colors = {
    "Dual expanded": "#9458a2",
    "Tumor singleton": "#ff8000",
    "NAT singleton": "#9cd0de",
    "Tumor multiplet": "#eeb3cb",
    "NAT multiplet": "#9cd0de",
    "Blood singleton": "#cce70b",
    "Blood multiplet": "#beac83"
}
```

```python
adata.obs["expansion_category"] = expansion_category
```

```python
# make categorical and store colors
adata._sanitize()
adata.uns["expansion_category_colors"] = [colors[x] for x in adata.obs["expansion_category"].cat.categories]
```

```python
ir.pl.clonal_expansion(adata, groupby="expansion_category")
```

```python
adata.obs["cell_type"] = adata.obs["cluster_orig"].str[0]
```

```python
adata.obs["tumor_type"] = adata.obs["patient"].str[:-1]
```

```python
sc.pl.umap(adata, color=["expansion_category", "cluster"], wspace=.3, size=5)
```

```python
ir.pl.group_abundance(
    adata, groupby="cluster", target_col="expansion_category", fraction=True
)
```

```python
ir.pl.group_abundance(
    adata, groupby="tumor_type", target_col="expansion_category", fraction=True
)
```

```python
ir.pl.group_abundance(
    adata, groupby="patient", target_col="blood_expanded", fraction=True
)
```

```python
%%time
adata_lung6 =adata[adata.obs["patient"] == "Lung6", :]
```

```python
fig, ax = subplots(2,2, figsize=(10, 8))
ir.pl.group_abundance(
    adata, groupby="blood_expanded", target_col="expansion_category", fraction=True
)
```

```python jupyter={"outputs_hidden": true}
for patient in ["Lung6", "Renal1", "Renal2", "Renal3"]:
    ir.pl.group_abundance(
        adata[adata.obs["patient"] == patient,:], groupby="blood_expanded", target_col="expansion_category", fraction=True)
```

```python

```

```python
adata.obs["expansion_category"].cat.categories
```

```python

```

```python
sc.pl.umap(adata, color=["expansion_category"], wspace=.3, size=5)
```

```python
adata.uns["expansion_category_cocatrs"]
```

```python

```
