---
jupyter:
  jupytext:
    formats: md,ipynb
    notebook_metadata_filter: -kernelspec
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
---

# Analysis of 140k T cells from cancer

<!-- #region raw_mimetype="text/restructuredtext" -->
In this notebook, we re-analyze single-cell TCR/RNA-seq data from [Wu et al. (2020)](https://www.nature.com/articles/s41586-020-2056-8)
generated on the 10x Genomics platform. The original dataset consists of >140k T cells
from 14 treatment-naive patients across four different types of cancer. Roughly 100k of the 140k cells have T-cell receptors.
<!-- #endregion -->

## 0. Setup

```python
%load_ext autoreload
%autoreload 2
import sys

sys.path.insert(0, "../../scirpy/")
import scirpy as ir
import pandas as pd
import numpy as np
import scanpy as sc
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib
from weblogo.seq import SeqList, unambiguous_protein_alphabet
from weblogo import png_formatter, svg_formatter, eps_formatter
from weblogo import LogoData, LogoOptions, LogoFormat
from IPython.display import Image, display
import warnings

sc.settings._vector_friendly = True

warnings.filterwarnings('ignore', category=FutureWarning)

# whether to run the alignment or use a cached version.
# If `False` and the cached version is not available, will raise an error.
run_alignment = False
```

This notebook was run with scirpy commit `e07553f6b84515ea2c48d2955660a3d6f900f9eb` and the following software versions:

```python
sc.logging.print_versions()
```

```python
def weblogo(seqs, title="", format="png"):
    """Draw a sequence logo from a list of amino acid sequences. """
    logodata = LogoData.from_seqs(SeqList(seqs, alphabet=unambiguous_protein_alphabet))
    logooptions = LogoOptions(logo_title=title)
    logoformat = LogoFormat(logodata, logooptions)
    if format == "png":
        display(Image(png_formatter(logodata, logoformat)))
    elif format == "eps":
        return eps_formatter(logodata, logoformat)
```

```python
# colors from the paper
colors = {
    "Tumor+NAT expanded": "#9458a2",
    "Tumor singleton": "#ff8000",
    "NAT singleton": "#f7f719",
    "Tumor multiplet": "#eeb3cb",
    "NAT multiplet": "#9cd0de",
    "Blood singleton": "#cce70b",
    "Blood multiplet": "#beac83",
}
```

## 1. Preparing the data

<!-- #region raw_mimetype="text/restructuredtext" -->
The dataset ships with the `scirpy` package. We can conveniently load it from the `scirpy.datasets` module.
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

```python
adata.obs.columns
```

```python
adata.obs["counts"] = adata.X.sum(axis=1).A1
```

### Preprocess Transcriptomics data

Transcriptomics data needs to be filtered and preprocessed as with any other single-cell dataset.
We recommend following the [scanpy tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
and the best practice paper by [Luecken et al.](https://www.embopress.org/doi/10.15252/msb.20188746).
For the _Wu et al. (2020)_ dataset, the authors already provide clusters and UMAP coordinates.
Instead of performing clustering and cluster annotation ourselves, we will just use
provided data. The clustering and annotation procedure used by the authors is described in their [paper](https://www.nature.com/articles/s41586-020-2056-8#Sec2). 

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
    "nan": "nan",
}
adata.obs["cluster"] = [mapping[x] for x in adata.obs["cluster_orig"]]
```

Let's inspect the UMAP plots. The first three panels show the UMAP plot colored by sample, patient and cluster.
We don't observe any clustering of samples or patients that could hint at batch effects.

The last three panels show the UMAP colored by the T cell markers _CD8_, _CD4_, and _FOXP3_.
We can confirm that the markers correspond to their respective cluster labels.

```python
sc.pl.umap(
    adata,
    color=["sample", "patient", "cluster", "CD8A", "CD4", "FOXP3"],
    ncols=2,
    wspace=0.5,
)
```

### TCR Quality Control

<!-- #region raw_mimetype="text/restructuredtext" -->
While most of T cell receptors have exactly one pair of α and β chains, up to one third of
T cells can have *dual TCRs*, i.e. two pairs of receptors originating from different alleles ([Schuldt et al (2019)](https://doi.org/10.4049/jimmunol.1800904)).

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
ax = ir.pl.group_abundance(
    adata, groupby="chain_pairing", target_col="has_tcr", normalize="has_tcr"
)
ax.set_ylabel("fraction of cells")
ax.set_xlabel("chain pairing")
ax.set_title("")
fig = ax.get_figure()
fig.savefig("figures/chain_pairing.svg")
```

```python
sc.pl.umap(
    adata,
    color="chain_pairing",
    groups=["Extra beta", "Extra alpha", "Two full chains"],
    size=[
        10 if x in ["Extra beta", "Extra alpha", "Two full chains"] else 3
        for x in adata.obs["chain_pairing"]
    ],
)
```

Indeed, in this dataset, ~7% of cells have more than one pair of productive T-cell receptors:

```python
print(
    "Fraction of cells with more than one pair of TCRs: {:.2f}".format(
        np.sum(
            adata.obs["chain_pairing"].isin(
                ["Extra beta", "Extra alpha", "Two full chains"]
            )
        )
        / adata.n_obs
    )
)
```

### Excluding multichain cells
Next, we visualize the _Multichain_ cells on the UMAP plot and exclude them from downstream analysis.
Multichain cells likely represent doublets. This is corroborated by the fact that they tend to
have a very high number of detected transcripts.

```python
_, pvalue = sp.stats.mannwhitneyu(
    adata.obs.loc[adata.obs["multi_chain"] == "True", "counts"].values,
    adata.obs.loc[adata.obs["multi_chain"] == "False", "counts"].values,
)
```

```python
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [2, 1]}
)
sc.pl.umap(
    adata,
    color="chain_pairing",
    groups="Multichain",
    size=[30 if x == "Multichain" else 3 for x in adata.obs["chain_pairing"]],
    ax=ax1,
    legend_loc="none",
    show=False,
    frameon=False,
    title="Multichain UMAP",
)
sc.pl.violin(
    adata, "counts", "multi_chain", ax=ax2, show=False, inner="box", stripplot=False
)
ax2.set_ylabel("detected reads")
ax2.set_xlabel("")
ax2.set_xticklabels(["other", "multichain"])
ax2.set_title("detected reads per cell")
ax2.text(0.3, 50000, f"p={pvalue:.2E}")
plt.subplots_adjust(wspace=0.5)
fig.savefig("figures/multichains.svg", dpi=600)
```

```python
print(
    f"Median counts Multichain: {np.median(adata.obs.loc[adata.obs['multi_chain'] == 'True', 'counts'].values)}"
)
print(
    f"Median counts other: {np.median(adata.obs.loc[adata.obs['multi_chain'] == 'False', 'counts'].values)}"
)
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

## 2. Define clonotypes

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
ir.pp.tcr_neighbors(
    adata, receptor_arms="all", dual_tcr="all",
)
```

```python
%%time
ir.tl.define_clonotypes(adata)
```

### identical amino acid sequences
Alternatively, we can define "clonotype clusters" based on identical amino acid sequences. 
Other than clonotypes, these do not necessarily arise from the same antecedent cell, but they recognize the same epitope. 

```python
%%time
ir.pp.tcr_neighbors(
    adata, receptor_arms="all", dual_tcr="all", metric="identity", sequence="aa"
)
```

```python
%%time
ir.tl.define_clonotype_clusters(adata, metric="identity", sequence="aa")
```

### similar amino acid sequences

With `scirpy`, it is possible to to one step further and summarize cells into the same clonotype cluster that might recognize the same epitope because they have *similar* amino acid sequences. 
This can be done by leveraging *Levenshtein* or *alignment distances*. 
Here, we compute the alignment distance with a cutoff of `15`, which is equivalent of three `A`s mutating into `R`.

```python
# adata.write_h5ad("./adata_in.h5ad")
```

```python
%%time
if run_alignment:
    ir.pp.tcr_neighbors(
        adata,
        receptor_arms="all",
        dual_tcr="all",
        cutoff=15,
        sequence="aa",
        metric="alignment",
        n_jobs=32
    )
    adata.write_h5ad("./adata_alignment.h5ad")
else:
    # read cached version
    adata = sc.read_h5ad("./adata_alignment.h5ad")
```

```python
%%time
ir.tl.define_clonotype_clusters(
    adata, metric="alignment", sequence="aa"
)
```

<!-- #region raw_mimetype="text/restructuredtext" -->
### Visualizing clonotype networks
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
ir.pl.clonotype_network(
    adata,
    color=["clonotype", "patient"],
    edges=False,
    size=50,
    ncols=2,
    legend_fontoutline=2,
    legend_loc=["on data", "right margin"],
)
```

## 3. Clonotype consistency

Before we dive into the analysis of clonal expansion, we compare the different approaches of clonotype definition.

### 3.1 nucleotide based (scirpy vs. Wu et al.)
In this section, we compare the clonotypes assigned by `scirpy` assigned to the clonotypes assigned by the authors of the study.
The original clonotypes are stored in the `clonotype_orig` column of `obs`.

According to the paper, the clonotypes are defined on nucleotide sequences and require sequence identity of all available chains.
This should be equivalent of running `scirpy.tcr_neighbors` with `dual_tcr="all"` and `receptor_arms="all"`.

To assess if the clonotype definitions are equivalent, we first check if there are *clonotypes in `clonotype_orig` that contain multiple clonotypes according to scirpy's definition*.

```python
# the clonotypes in clonotype_orig that contain multiple clonotype_nt
nt_in_orig = (
    adata.obs.groupby(["clonotype_orig", "clonotype"], observed=True)
    .size()
    .reset_index()
    .groupby("clonotype_orig")
    .size()
    .reset_index()
)
nt_in_orig[nt_in_orig[0] > 1]
```

There are none!

Next, we check if there are *clonotypes in `clonotype_nt` that contain multiple clonotypes according to the authors' definition.*

```python
# the clonotypes in clonotype_nt that contain multiple clonotype_orig
orig_in_nt = (
    adata.obs.groupby(["clonotype", "clonotype_orig"], observed=True)
    .size()
    .reset_index()
    .groupby("clonotype")
    .size()
    .reset_index()
)
with pd.option_context("display.max_rows", 8):
    display(orig_in_nt[orig_in_nt[0] > 1])
```

There are a few!

Let's investigate that further:

```python
with pd.option_context("display.max_rows", 10):
    display(
        adata.obs.loc[
            adata.obs["clonotype"].isin(
                orig_in_nt[orig_in_nt[0] > 1]["clonotype"]
            ),
            [
                "clonotype",
                "clonotype_orig",
                "patient",
                "TRA_1_cdr3_nt",
                "TRA_2_cdr3_nt",
                "TRB_1_cdr3_nt",
                "TRB_2_cdr3_nt",
            ],
        ].sort_values(["clonotype", "clonotype_orig"])
    )
```

All cells of the same `clonotype_nt` have the same nucleotide sequences (apart from swapping between `TRA_1` and `TRA_2` or `TRB_1` and `TRB_2`, respectively). Our method appears to work as expected.
The inconsistencies seem to arise from the fact that the clonotypes have the same sequences, but originate from different patients. Apparently, the
authors did not allow clonotypes from different patients (which makes sense when approaching clonotypes from a genomic point of view, not from an epitope recognition point of view).

When checking the number of clonotypes_orig per clonotype_nt and patient, we eradicate the differences:

```python
# the clonotypes in clonotype_nt that contain multiple clonotype_orig
orig_in_nt = (
    adata.obs.groupby(["clonotype", "patient", "clonotype_orig"], observed=True)
    .size()
    .reset_index()
    .groupby(["clonotype", "patient"])
    .size()
    .reset_index()
)
with pd.option_context("display.max_rows", 8):
    display(orig_in_nt[orig_in_nt[0] > 1])
```

Now, also the number of clonotypes is consistent:

```python
assert (
    adata.obs["clonotype_orig"].unique().size
    == adata.obs.groupby(["patient", "clonotype"], observed=True).size().reset_index().shape[0]
)
print(
    f"""Number of clonotypes according to Wu et al.: {adata.obs["clonotype_orig"].unique().size}"""
)
print(
    f"""Number of clonotypes according to scirpy: {adata.obs.groupby(["clonotype"]).size().reset_index().shape[0]}"""
)
print(
    f"""Number of clonotypes according to scirpy, within patient: {adata.obs.groupby(["patient", "clonotype"], observed=True).size().reset_index().shape[0]}"""
)
```

### 3.2 amino-acid vs. nucleotide-based


For a few clonotypes, we observe differences comparing the amino-acid vs. nucleotide-based approach for defining clonotypes.
Clonotypes with the same amino acid sequence but different nucleotide sequences recognize the same antigen -
but derive from different antedescent cells. This can be an example of convergent evolution.

```python
ir.tl.clonotype_network(adata, sequence='aa', metric='identity', min_size=50)
```

```python
ir.pl.clonotype_network(
    adata,
    color=["ct_cluster_aa_identity", "clonotype"],
    edges=False,
    size=50,
    ncols=2,
    legend_fontoutline=2,
    legend_loc=["on data", "none"],
)
```

Let's find the cells that belong to clonotypes that have different nucleotide sequences but the same amino acid sequences:

```python
nt_in_aa = (
    adata.obs.groupby(["ct_cluster_aa_identity", "clonotype"], observed=True)
    .size()
    .reset_index()
    .groupby(["ct_cluster_aa_identity"])
    .size()
    .reset_index()
)
convergent_clonotypes = nt_in_aa.loc[nt_in_aa[0] > 1, "ct_cluster_aa_identity"]
```

```python
adata.obs["is_convergent"] = [
    "convergent" if x else "-"
    for x in adata.obs["ct_cluster_aa_identity"].isin(convergent_clonotypes)
]
```

```python
sc.pl.umap(
    adata,
    color="is_convergent",
    groups=["convergent"],
    size=[10 if x == "convergent" else 3 for x in adata.obs["is_convergent"]],
)
```

The phonomenon appears to primarily occur in CD8+ effector and tissue resident T cells:

```python
ir.pl.group_abundance(
    adata, groupby="cluster", target_col="is_convergent", normalize=True
)
```

### 3.3 amino-acid identity vs. alignment distance
When computing the alignment distance, we allowed a distance of `15`, based on the BLOSUM62 matrix.
This is eqivalent of three `A`s mutating into `R`.

The alignment distance allows us to identify similar CDR3 sequences, that likely recognize the same epitope.

For the following visualization, we exclude cells with *Orphan* chains. Having only one chain that
needs to match there tend to be a lot of similar cells within a distance of `15` resulting in
large, uninformative networks for those cells.

```python
%%time
adata_sub = adata[~adata.obs["chain_pairing"].str.startswith("Orphan"), :]
```

```python
%%time
ir.tl.clonotype_network(
    adata_sub,
    min_size=50,
    layout="components",
    sequence="aa",
    metric="alignment"
)
```

We'll have a closer look at the following clonotypes, as they seem interesting:

```python
selected_clonotypes = ["1312", "5499", "1688", "644"]
```

```python
ir.pl.clonotype_network(
    adata_sub,
    color=["ct_cluster_aa_alignment"],
    groups=selected_clonotypes,
    edges=False,
    size=50,
    ncols=2,
    legend_loc="on data",
    legend_fontoutline=3,
)
```

```python
np.random.seed(123)
np.random.shuffle(adata_sub.uns["clonotype_colors"])
```

```python
ir.pl.clonotype_network(
    adata_sub,
    color=["clonotype", "patient"],
    edges=False,
    size=50,
    ncols=2,
    legend_loc=["none", "right margin"],
    legend_fontoutline=2,
)
```

```python
%%capture
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ir.pl.clonotype_network(
    adata_sub,
    color="clonotype",
    edges=False,
    size=50,
    legend_loc="none",
    ax=ax,
    frameon=False,
)
ax.set_title("clonotypes")
fig.savefig("figures/clonotype_network.svg")
```

The patient distribution of the four clonotypes:

```python
ax = ir.pl.group_abundance(
    adata_sub[adata_sub.obs["ct_cluster_aa_alignment"].isin(selected_clonotypes), :],
    groupby="ct_cluster_aa_alignment",
    sort=selected_clonotypes,
    target_col="patient",
)
ax.xaxis.set_tick_params(rotation=0)
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("center")
ax.tick_params(labelsize=10)
fig = ax.get_figure()
ax.set_title("")
ax.set_xlabel("clonotype")
fig.savefig("figures/clonotype_network_patients.svg")
```

#### Sequence logos

```python
for ct in selected_clonotypes:
    for chain, chain_label in zip(["TRA_1_cdr3", "TRB_1_cdr3"], ["alpha", "beta"]):
        with open(f"figures/logo_ct_{ct}_{chain_label}.eps", "wb") as f:
            f.write(
                weblogo(
                    adata.obs.loc[adata.obs["ct_cluster_aa_alignment"] == ct, chain].values,
                    title=f"clonotype {ct}: {chain_label}-chain",
                    format="eps",
                )
            )
```

```python
for ct in selected_clonotypes:
    for chain, chain_label in zip(["TRA_1_cdr3", "TRB_1_cdr3"], ["alpha", "beta"]):
        weblogo(
            adata.obs.loc[adata.obs["ct_cluster_aa_alignment"] == ct, chain].values,
            title=f"clonotype {ct}: {chain_label}-chain",
            format="png",
        )
```

Clonotype `644` is shared among multiple patients and could be specific for a common, viral antigen.

In case of the clonotype `1312` we find evidence in *vdjdb* that it could be specific for an Human Cytomegalievirus (CMV) antigen:
 * In [vdjdb](https://vdjdb.cdr3.net/search), we find a CMV-specific alpha-chain searching with the `CAV[STR][LG]QAGTALIF` pattern.
 * Searching for the beta-chain pattern does not yield a direct result. However, allowing up to two substitutions, we find a CMV-specific chain as well.


#### convergent clonotypes
We define a clonotype as being convergent, if there are different versions of nucleotide sequences for similar clonotypes, within the same patient.

We define convergence
 * on the level of amino-acid identity vs nucleotide identity
 * on the level of amino-acid similarity vs nucleotide identity


```python
convergent_aa = (
    adata.obs.groupby(["ct_cluster_aa_identity", "ct_cluster_aa_alignment", "patient"], observed=True)
    .size()
    .reset_index()
    .groupby(["ct_cluster_aa_alignment", "patient"], observed=True)
    .size()
    .reset_index()
)
convergent_nt = (
    adata.obs.groupby(["clonotype", "ct_cluster_aa_alignment", "patient"], observed=True)
    .size()
    .reset_index()
    .groupby(["ct_cluster_aa_alignment", "patient"], observed=True)
    .size()
    .reset_index()
)
convergent_clonotypes_aa = convergent_aa.loc[
    convergent_aa[0] > 1, "ct_cluster_aa_alignment"
].values
convergent_clonotypes_nt = convergent_nt.loc[
    convergent_nt[0] > 1, "ct_cluster_aa_alignment"
].values
```

```python
# convergent clonotypes, AA identity vs alignment
adata.obs["convergent_aa"] = [
    str(x)
    for x in adata.obs["ct_cluster_aa_alignment"].isin(convergent_clonotypes_aa)
    & ~adata.obs["chain_pairing"].str.startswith("Orphan")
]
# convergent clonotypes, NN identity vs AA alignment
adata.obs["convergent_nt"] = [
    str(x)
    for x in adata.obs["ct_cluster_aa_alignment"].isin(convergent_clonotypes_nt)
    & ~adata.obs["chain_pairing"].str.startswith("Orphan")
]
```

```python
sc.pl.umap(
    adata,
    color=["convergent_aa"],
    groups="True",
    size=[10 if x == "True" else 3 for x in adata.obs["convergent_aa"]],
)
```

```python
sc.pl.umap(
    adata,
    color=["convergent_nt"],
    groups="True",
    size=[10 if x == "True" else 3 for x in adata.obs["convergent_nt"]],
)
```

```python
%%capture
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
sc.pl.umap(
    adata,
    color=["convergent_nt"],
    groups="True",
    size=[10 if x == "True" else 3 for x in adata.obs["convergent_nt"]],
    ax=ax,
    frameon=False,
    legend_loc="none",
    show=False,
)
ax.set_title("convergent clonotypes")
fig.savefig("figures/convergent_clonotypes_umap.svg")
```

```python
ir.pl.group_abundance(
    adata,
    groupby="cluster_orig",
    target_col="convergent_nt",
    normalize=True,
    sort="alphabetical",
    fig_kws={"dpi": 120},
)
```

#### Inspecting only converged clonotypes

```python
adata_converged = adata[
    (adata.obs["convergent_nt"] == "True")
    & ~adata.obs["chain_pairing"].str.startswith("Orphan"),
    :,
]
```

```python
%%time
ir.tl.clonotype_network(
    adata_converged,
    min_size=1,
    layout="components",
    metric="alignment",
    sequence="aa"
)
```

```python
ir.pl.clonotype_network(
    adata_converged,
    color=["ct_cluster_aa_alignment", "clonotype"],
    edges=False,
    size=50,
    ncols=2,
    legend_loc=["none", "none"],
    legend_fontoutline=2,
)
```

## 4. Clonal expansion
In this section, we assess the clonal expansion

 - across patients
 - across clusters
 - across tissue sources

With the `pl.clonal_expansion` function, we can easily visualize the clonal expansion
by different variables. Per default, the plot shows the fraction of
cells belonging to an expanded clonotype.

### clonal expansion across patients

```python
fig, ax = plt.subplots(dpi=100)
ir.pl.clonal_expansion(
    adata, groupby="patient", summarize_by="cell", show_nonexpanded=True, ax=ax
)
ax.tick_params(labelsize=10)
fig.savefig("figures/expansion_patients_cell.svg")
```

Alternatively, we can show the *fraction of expanded clonotypes*.

```python
fig, ax = plt.subplots(dpi=100)
ir.pl.clonal_expansion(
    adata,
    groupby="patient",
    summarize_by="clonotype",
    show_nonexpanded=False,
    color=["#FF7F0E", "#2CA02C"],
    ax=ax,
)
ax.tick_params(labelsize=10)
fig.savefig("figures/expansion_patients_clonotype.svg")
```

In the paper, the authors state that depending on the patient, between 9 and 18% of clonotypes
are expanded. Our results are highly consistent with these results:

```python
fraction_expanded = (
    ir.tl.summarize_clonal_expansion(
        adata, groupby="patient", summarize_by="clonotype", normalize=True
    )
    .drop("1", axis="columns")
    .sum(axis=1)
)
```

```python
print(
    f"Between {min(fraction_expanded):.1%} and {max(fraction_expanded):.1%} of clonotypes are expanded."
)
```

### clonal expansion across cell-type clusters
We observe that clonal expansion is highest among the CD8+ T-cell clusters, in particular effector and tissue-resident T cells.

```python
ir.pl.clonal_expansion(
    adata,
    groupby="cluster_orig",
    summarize_by="cell",
    show_nonexpanded=True,
    fig_kws={"dpi": 120},
)
```

### clonal expansion across tissue sources
The fraction of expanded cell is roughly equivalent among tumor and ajacent normal tissue, but lower in
blood.

```python
ir.pl.clonal_expansion(
    adata, groupby="source", summarize_by="cell", show_nonexpanded=True
)
```

## 5. Dual- and blood expanded clonotypes
Finally, we will divide the clonotypes into different categories, based on their expansion in blood, tissue and tumor samples.

In particular, we will
 * identify *dual-expanded* clonotypes, which are expanded in both adjacent tissue and tumor samples
 * identify blood-expanded clonotypes

and compare their abundance across cell-types and patients.

For an illustration, please refer to Fig. 1b of the *Wu et al. (2020)* paper.

```python
clonotype_size_by_source = (
    adata.obs.groupby(["patient", "source", "clonotype"], observed=True)
    .size()
    .reset_index(name="clonotype_count_by_source")
)
adata.obs = (
    adata.obs.reset_index()
    .merge(clonotype_size_by_source, how="left")
    .set_index("index")
)
```

```python
blood_expanded = []
for is_expanded, source in zip(
    (adata.obs["source"] == "Blood") & (adata.obs["clonotype_count_by_source"] > 1),
    adata.obs["source"],
):
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
clonotype_membership = {ct: list() for ct in adata.obs["clonotype"]}
for clonotype, source in zip(adata.obs["clonotype"], adata.obs["source"]):
    clonotype_membership[clonotype].append(source)
clonotype_membership = {
    ct: set(sources) for ct, sources in clonotype_membership.items()
}
```

```python
expansion_category = []
for clonotype, clonotype_size, source in zip(
    adata.obs["clonotype"], adata.obs["clonotype_size"], adata.obs["source"]
):
    if clonotype_size == 1:
        if source == "Blood":
            expansion_category.append("Blood singleton")
        elif source == "NAT":
            expansion_category.append("NAT singleton")
        elif source == "Tumor":
            expansion_category.append("Tumor singleton")
    elif clonotype_size > 1:
        membership = clonotype_membership[clonotype]
        if "Tumor" in membership and "NAT" in membership:
            expansion_category.append("Tumor+NAT expanded")
        elif "Tumor" in membership:
            expansion_category.append("Tumor multiplet")
        elif "NAT" in membership:
            expansion_category.append("NAT multiplet")
        elif "Blood" in membership:
            # these are *only* expanded in blood
            expansion_category.append("Blood multiplet")

assert len(expansion_category) == adata.n_obs
```

```python
adata.obs["cell_type"] = adata.obs["cluster_orig"].str[0]
adata.obs["tumor_type"] = adata.obs["patient"].str[:-1]
```

```python
adata.obs["expansion_category"] = expansion_category
```

```python
# make categorical and store colors
adata._sanitize()
adata.uns["expansion_category_colors"] = [
    colors[x] for x in adata.obs["expansion_category"].cat.categories
]
```

Mostly CD8+ effector and tissue-resident T cells are blood-expanded and dual-expanded:

```python
sc.pl.umap(
    adata,
    color="blood_expanded",
    groups=["expanded", "not expanded"],
    size=[
        15 if x in ["expanded", "not expanded"] else 3
        for x in adata.obs["blood_expanded"]
    ],
)
```

```python
ir.pl.embedding(
    adata,
    basis="umap",
    color=["expansion_category", "cluster_orig"],
    legend_loc=["right margin", "on data"],
    show=True,
    legend_fontoutline=2,
    frameon=False,
    wspace=0.5,
    panel_size=(6, 4),
)
```

```python
%%capture
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4), dpi=300)
sc.pl.umap(
    adata,
    color="expansion_category",
    legend_loc="none",
    show=False,
    ax=ax1,
    frameon=False,
)
sc.pl.umap(
    adata,
    color="cluster_orig",
    legend_loc="on data",
    show=False,
    ax=ax2,
    legend_fontoutline=2,
    frameon=False,
)
ax1.set_title("clonal expansion pattern")
ax2.set_title("cell-type cluster")
plt.subplots_adjust(wspace=0.1)
fig.savefig("figures/clonal_expansion_umap.svg")
```

```python
%%capture
fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
sc.pl.umap(
    adata,
    color="cluster_orig",
    legend_loc="none",
    show=False,
    ax=ax,
    legend_fontoutline=2,
    frameon=False,
)
ax.set_title("")
fig.savefig("figures/cluster_orig.png")
```

### Repertoire overlap

```python
ir.tl.repertoire_overlap(adata, groupby="sample", target_col="ct_cluster_aa_alignment")
ax = ir.pl.repertoire_overlap(adata, groupby="sample", heatmap_cats=["patient", "tumor_type"])
ax.fig.savefig("figures/repertoire_overlap.svg")
```

### Categories by cell type (fractions and count):

```python
fig, ax = plt.subplots(dpi=120)
ir.pl.group_abundance(
    adata,
    groupby="cluster_orig",
    target_col="expansion_category",
    normalize=True,
    ax=ax,
    sort="alphabetical",
)
ax.set_title("")
ax.set_xlabel("cluster")
fig.savefig("figures/expansion_category_cluster_norm.svg")
```

```python
fig, ax = plt.subplots(dpi=120)
ir.pl.group_abundance(
    adata,
    groupby="cluster_orig",
    target_col="expansion_category",
    normalize=False,
    ax=ax,
    sort="alphabetical",
)
ax.set_title("")
ax.set_xlabel("cluster")
fig.savefig("figures/expansion_category_cluster.svg")
```

### Categories by tumor type:

```python
ir.pl.group_abundance(
    adata, groupby="tumor_type", target_col="expansion_category", normalize=True
)
```

The fraction of blood-expanded, blood non-expanded and blood-independent cells
for the four patients with blood samples

```python
ir.pl.group_abundance(
    adata[adata.obs["patient"].isin(["Lung6", "Renal1", "Renal2", "Renal3"]), :],
    groupby="patient",
    target_col="blood_expanded",
    normalize=True,
)
```

### Tissue infiltration patterns. 

* The bar plots show the distribution of cells by tissue expansion patterns from
blood-indpendned, non-expanded and expanded clones.

```python
for patient in ["Lung6", "Renal1", "Renal2", "Renal3"]:
    ax = ir.pl.group_abundance(
        adata[adata.obs["patient"] == patient, :],
        groupby="blood_expanded",
        target_col="expansion_category",
        normalize=True,
    )
    ax.set_title(patient)
```

### Tissue expansion patterns of T cell by cluster.

```python
for subset in [
    ["NAT singleton", "NAT multiplet"],
    ["Tumor singleton", "Tumor multiplet"],
]:
    ax = ir.pl.group_abundance(
        adata[adata.obs["expansion_category"].isin(subset), :],
        groupby="cluster_orig",
        target_col="expansion_category",
        normalize=False,
        fig_kws={"dpi": 120},
        sort="alphabetical",
    )
    ax.set_title(subset[0].split()[0])
```

```python
for source in ["NAT", "Tumor"]:
    ax = ir.pl.group_abundance(
        adata[
            (adata.obs["source"] == source)
            & (adata.obs["expansion_category"] == "Tumor+NAT expanded"),
            :,
        ],
        groupby="cluster_orig",
        target_col="expansion_category",
        normalize=False,
        fig_kws={"dpi": 120},
        sort="alphabetical",
    )
    ax.set_title(source)
```

## 6. Gene usage

```python
ax = ir.pl.vdj_usage(adata, full_combination=False, top_n=30)
ax.set_ylabel("cells")
ax.set_xlabel("VDJ segment")
ax.set_title("VDJ usage")
fig = ax.get_figure()
fig.savefig("figures/vdj_usage.svg")
```

```python
ax = ir.pl.spectratype(
    adata[adata.obs["TRB_1_j_gene"] != "None", :],
    cdr3_col="TRB_1_cdr3",
    color="TRB_1_j_gene",
    fig_kws={"dpi": 120},
)
ax.xaxis.set_tick_params(rotation=90)
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("center")
ax.set_title("Spectratype of primary TCR-β chain")
ax.set_xlabel("CDR3 sequence length")
fig = ax.get_figure()
fig.savefig("figures/spectratype.svg")
```

```python

```
