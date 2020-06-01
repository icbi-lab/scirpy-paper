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

# Analysis of 3k T cells from cancer

<!-- #raw raw_mimetype="text/restructuredtext" -->

In this tutorial, we re-analize single-cell TCR/RNA-seq data from Wu et al (:cite:`Wu2020`)
generated on the 10
x Genomics VDJ platform. The original dataset consists of >140k T cells
from 14 treatment-naive patients across four different types of cancer.
For this tutorial, to speed up computations, we use a downsampled version of 3k cells.

<!-- #endraw -->

```python
%load_ext autoreload
%autoreload 2
import sys

sys.path.append("../..")
import scirpy as ir
import pandas as pd
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
```

<!-- #raw raw_mimetype="text/restructuredtext" -->

The dataset ships with the `scirpy` package. We can conveniently load it from the `dataset` module:

<!-- #endraw -->

```python
adata = ir.datasets.wu2020_3k()
```

<!-- #raw raw_mimetype="text/restructuredtext" -->

`adata` is a regular :class:`~anndata.AnnData` object:

<!-- #endraw -->

```python
adata.shape
```

It just has additional TCR-related columns in `obs`:

```python
adata.obs
```

<!-- #raw raw_mimetype="text/restructuredtext" -->

.. note:: **Importing data**

`scirpy` supports importing TCR data from Cellranger or `TraCeR <https://github.com/Teichlab/tracer>`\_.
See :ref:`api-io` for more details.

This particular dataset has been imported using :func:`scirpy.read_10x_vdj_csv` and merged
with transcriptomics data using :func:`scirpy.pp.merge_with_tcr`. The exact procedure
is described in :func:`scirpy.datasets.wu2020`.

<!-- #endraw -->

## Preprocess Transcriptomics data

Transcriptomics data needs to be filtered and preprocessed as with any other single-cell dataset.
We recommend following the [scanpy tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html)
and the best practice paper by [Luecken et al.](https://www.embopress.org/doi/10.15252/msb.20188746)

```python
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_cells(adata, min_genes=100)
```

```python
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1000)
sc.pp.log1p(adata)
```

For the _Wu2020_ dataset, the authors already provide clusters and UMAP coordinates.
Instead of performing clustering and cluster annotation ourselves, we will just use
provided data.

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

While most T cell receptors have exactely one pair of α and β chains, it has been reported that up to one third of
the receptors can be _Dual TCRs_ posessing a second pair of receptors, one originating from each Allele (:cite:`Schuldt2019`).

Using the :func:`scirpy.tl.chain_pairing` function, we can add a summary
about the T cell receptor compositions to `adata.obs`. We can visualize it using :func:`scirpy.pl.group_abundance`.

.. note:: **chain pairing**

- _Orphan chain_ refers to cells that have a single alpha or beta receptor chain, respectively.
- _Extra chain_ refers to cells that have a full alpha/beta receptor pair, and an additional chain.
- _Multichain_ refers to cells with more than two receptor pairs detected. These cells are likely doublets.
  <!-- #endraw -->

```python
ir.tl.chain_pairing(adata)
```

```python
ir.pl.group_abundance(
    adata, groupby="chain_pairing", target_col="source",
)
```

Indeed, in our case, ~20% of cells have more than a one pair of T-cell receptors:

```python
print("Fraction of cells with more than one pair of TCRs: {:.2f}".format(
    np.sum(adata.obs["chain_pairing"].isin(["Extra beta", "Extra alpha", "Two full chains"])) / adata.n_obs
))
```

Next, we visualize the _Multichain_ cells on the UMAP plot and exclude them from downstream analysis:

```python
sc.pl.umap(adata, color="multi_chain")
```

```python
adata = adata[adata.obs["multi_chain"] != "True", :].copy()
```

## Define clonotypes

<!-- #raw raw_mimetype="text/restructuredtext" -->

In this section, we will define and visualize clonotypes.

_Scirpy_ implements a network-based approach for clonotype definition. The steps to
create and visualize the clonotype-network are anologous to the construction of a
neighborhood graph we already know from transcriptomics data:

.. list-table:: Analysis steps on transcriptomics data
:widths: 40 60
:header-rows: 1

- - scanpy function
  - objective
- - :func:`scanpy.pp.neighbors`
  - Compute a nearest-neighbor graph based on gene expression.
- - :func:`scanpy.tl.leiden`
  - Cluster cells by the similarity of their transcriptional profiles.
- - :func:`scanpy.tl.umap`
  - Compute positions of cells in UMAP embedding.
- - :func:`scanpy.pl.umap`
  - Plot UMAP colored by different parameters.

.. list-table:: Analysis steps on TCR data
:widths: 40 60
:header-rows: 1

- - scirpy function
  - objective
- - :func:`scirpy.pp.tcr_neighbors`
  - Compute a neighborhood graph of CDR3-sequences.
- - :func:`scirpy.tl.define_clonotypes`
  - Cluster cells by the similarity of their CDR3-sequences.
- - :func:`scirpy.tl.clonotype_network`
  - Compute positions of cells in clonotype network.
- - :func:`scirpy.pl.clonotype_network`
  - Plot clonotype network colored by different parameters.

<!-- #endraw -->

### Compute CDR3 neighborhood graph

<!-- #raw raw_mimetype="text/restructuredtext" -->

:func:`scirpy.pp.tcr_neighbors` computes pairwise sequence alignments of all CDR3 sequences and
derives a distance from the alignment score. This approach was originally proposed as _TCRdist_ by Dash et al. (:cite:`TCRdist`).

The function requires to specify a `cutoff` parameter. All cells with a distance between their
CDR3 sequences lower than `cutoff` will be connected in the network. In the first example,
we set the cutoff to `0`, to define clontypes as cells with **identical** CDR3 sequences.

Then, the function :func:`scirpy.tl.define_clonotypes` will detect connected modules
in the graph and annotate them as clonotypes. This will add a `clonotype` and
`clonotype_size` column to `adata.obs`.

<!-- #endraw -->

```python
ir.pp.tcr_neighbors(adata, strategy="all", chains="primary_only", cutoff=0)
ir.tl.define_clonotypes(adata)
```

<!-- #raw raw_mimetype="text/restructuredtext" -->

To visualize the network we first call :func:`scirpy.tl.clonotype_network` to compute the layout.
We can then visualize it using :func:`scirpy.pl.clonotype_network`. We recommend setting the
`min_size` parameter to `>=2`, to prevent the singleton clonotypes from cluttering the network.

<!-- #endraw -->

```python
ir.tl.clonotype_network(adata, min_size=2)
ir.pl.clonotype_network(adata, color="clonotype", legend_loc="none")
```

Let's re-compute the network with a `cutoff` of `20`.
That's the equivalent of 4 `R`s mutating into `N` (using the BLOSUM62 distance matrix).

Additionally, we set `chains` to `all`. This results in the distances not being only
computed between the most abundant pair of T-cell receptors, but instead, will
take the minimal distance between any pair of T-cell receptors.

```python
ir.pp.tcr_neighbors(adata, cutoff=20, chains="all")
ir.tl.define_clonotypes(adata)
```

```python
ir.tl.clonotype_network(adata, min_size=3)
```

When coloring by clonotype, we can see that the large, connected Hairball has been sub-divided in multiple clonotypes by
Graph-based clustering using the _Leiden-algorithm_. Also, the edges are now colored according to the distance
between nodes. The darker an edge, the lower the alignment-distance.

```python
ir.pl.clonotype_network(adata, color="clonotype", legend_fontoutline=2)
```

Now we show the same graph, colored by sample.
We observe that for instance clonotypes 292 and 279 are _private_, i.e. they contain cells from
a single sample only. On the other hand, for instance clonotype 16 is _public_, i.e.
it is shared across tissues and/or patients.

```python
ir.pl.clonotype_network(adata, color="sample")
```

Next, visualize the clonal expansion by cell-type cluster

```python
ir.pl.clonal_expansion(adata, groupby="cluster", clip_at=4, fraction=False)
```

Normalized to the cluster size

```python
ir.pl.clonal_expansion(adata, "cluster")
```

```python
ir.pl.alpha_diversity(adata, groupby="cluster")
```

### Clonotype abundance

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="cluster", max_cols=10, fraction=False
)
```

Perhaps an even more straightforward question would be comparing clonotype composition of samples

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="sample", max_cols=10, stacked=False
)
```

If cell types are considered, it is still probably better to normalize to cell numbers in a sample.

```python
ir.pl.group_abundance(
    adata, groupby="clonotype", target_col="cluster", fraction="sample", max_cols=10
)
```

Once we have information on clonotypes, we can also plot them on the UMAP


## Gene usage

Group abundance plots can also give some information on VDJ usage

```python
ir.pl.group_abundance(
    adata,
    groupby="TRB_1_v_gene",
    target_col="cluster",
    fraction="sample",
    max_cols=10,
    fig_kws={"dpi": 170},
)
```

```python
vdj_usage = ir.tl.group_abundance(
    adata, groupby="cluster", target_col="TRB_1_v_gene", fraction=True
)
```

```python
vdj_usage = vdj_usage.loc[:, ["TRBV20-1", "TRBV7-2", "TRBV28", "TRBV5-1", "TRBV7-9"]]
```

```python
ir.pl.base.bar(vdj_usage)
```

### Spectratype plots

```python
ir.pl.spectratype(adata, target_col="cluster",     fig_kws={"dpi": 170},)
```

```python
ir.pl.spectratype(adata, target_col="cluster", fraction="sample", viztype="line")
```

```python
ir.pl.spectratype(adata, target_col="cluster", fraction=False, viztype="line")
```

```python
ir.pl.spectratype(
    adata, groupby="TRB_1_cdr3_len", target_col="TRB_1_v_gene", fraction="sample", fig_kws={'dpi': 150}
)
```

```python
ir.pl.vdj_usage(adata)
```

```python
ir.pl.vdj_usage(adata, top_n=None, draw_bars=False)
```

```python
ir.pl.vdj_usage(adata, full_combination=False, top_n=None)
```

```python
ir.pl.vdj_usage(adata, full_combination=False, top_n=30)
```

 subsetting adata: ir.pl.vdj_usage(adata[for_cells, :], ...)

```python
ir.pl.vdj_usage(adata[adata.obs['sample'] == 'LT1'])
```

# Brainstorming


## Clonotype information on UMAP


Thinking with the head of future end users (wet lab researcher, focusing on the biology) I would anticipate a UMAP with the most abundant clonotypes and perhaps the V gene segments might be welcome.

```python
# This is probably hacky, we might think about a better way, but we need the most abundant clonotypes

top_clonotypes = adata.obs.clonotype.value_counts()[:8].index.values.tolist() # A better way might be needed especailly to take normalization into account
top_vgenes = adata.obs.TRB_1_v_gene.value_counts()[:8].index.values.tolist()
top_vgenes
```

```python
sc.pl.umap(adata, color="clonotype", groups=top_clonotypes)
```

```python
sc.pl.umap(adata, color="TRB_1_v_gene", groups=top_vgenes)
```

```python
# The multipanel version does not work this way unfortuanately
sc.pl.umap(adata, color=["clonotype", "source", "cluster", "TRB_1_v_gene"], groups=[top_clonotypes, None, None, top_vgenes], ncols=2, wspace=.5)
```

## Clonotype size and expansion for samples


Scanpy has the dotplot with dendrogram for genes, which I find pretty cool. I would imagine it a good chance to get some overview on the whole dataset to show samples, top clonotype, the size of those clonotypes and either the expansion or the publicness/privateness of that clonotype (later, when we have a nice measure for that)

```python
# This would be the plot I want to reimplement, maybe without the denrogram now

sc.tl.dendrogram(adata, groupby='patient')
markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
sc.pl.dotplot(adata, markers, groupby='patient', dendrogram=True)
```

```python
# Subset adata for the top clonotypes

samples = adata.obs['sample'].unique()
clonotypes = adata.obs.clonotype.value_counts()[:25].index.values.tolist()
N, M = len(samples), len(clonotypes)
df = adata.obs.loc[adata.obs.clonotype.isin(clonotypes), ['clonotype', 'sample']].copy()
df['clonotype'] = df.clonotype.astype(str) # If clonotype column is categorical, it inherits all the categories, not only the top few -> compicates groupby
# More on the problem with categorical grouping: https://github.com/pandas-dev/pandas/issues/17594

df
```

```python
# We need weight column for aggregation

df['normalized_weight'] = 1
df = df.groupby(['clonotype', 'sample']).agg({'normalized_weight': 'sum'}).reset_index()
df['normalized_weight'] = df['normalized_weight'].fillna(0)

df
```

```python
# Try to order clonotypes reasonably

df['clonotype_rank'] = df['clonotype'].map(dict(list(zip(*[clonotypes, range(len(clonotypes))]))))
df = df.sort_values(by='clonotype_rank', ascending=False)
df
```

```python
# A little distraction, but color the background based on sample source

colordict = dict()
source_types = adata.obs.source.unique()
n_color = len(source_types)
tile_cols = int(len(plt.rcParams['axes.prop_cycle'].by_key()['color'])/n_color)
tile_cols = np.tile(plt.rcParams['axes.prop_cycle'].by_key()['color'], tile_cols+1)
for i in range(n_color):
    colordict[source_types[i]] = tile_cols[i]
print(colordict)
    
habillage = adata.obs.groupby(['sample', 'source']).agg('size').reset_index()
habillage = habillage.set_index('sample')
habillage = habillage['source'].to_dict()
```

```python
# Finally, add a measure for clonal expansion that will influence the color

rcm = plt.get_cmap('Reds')
dot_colors = ir.tl.clip_and_count(adata, "sample", target_col="clonotype").iloc[:,2].apply(lambda x: rcm(x+0.7))
dot_colors = np.tile(dot_colors, M) # This is cheating a bit, since now every dot has the same color in a column and it is wasting the opportunity to show more information
```

```python
# Draw a dotplot

x = np.tile(range(N), M)
y = np.repeat(range(M), N)
sizes = df['normalized_weight'].tolist()
fig, ax = plt.subplots()
ax.set_xticks(range(N))
ax.set_xticklabels(samples, rotation=30, ha='right')
ax.set_xlabel('Sample')
ax.set_yticks(range(M))
ax.set_yticklabels(clonotypes[::-1])
ax.set_ylabel('Clonotype')
ax.scatter(x, y, color=dot_colors, s=sizes)
for i in range(N):
    ax.axvspan(i-0.3, i+0.3, color=colordict[habillage[samples[i]]], alpha=0.1)
```

It would be even cooler if we had a measure for activation (a set of genes, like IL-2 and their mean expression?) and the color would correspond to how much activated (or anergic?) the cells in each clonotype are.


## Repertoire overlap


If we are talking about dendrograms (and it would be nice to order samples based on it), we first need a pairwise similarity measure based on repertoire overlap. So first we need a tool for repertoire overlap.

```python
# The similarity measure could be Jaccard index. Of course this import here is a temporary solution

from scipy.spatial import distance as sp_dist
from scipy.cluster import hierarchy as sp_dendro
import seaborn as sns
```

```python
# Create a table of clonotype presence

df = adata.obs.loc[:, ['clonotype', 'sample']].copy()
df = df.groupby(['clonotype', 'sample']).agg('size').reset_index()
df['membership'] = 1
df = df.loc[:, ['clonotype', 'sample', 'membership']]
df = df.pivot(index='sample', columns='clonotype', values='membership')
df = df.fillna(0
```

```python
# Compute distances with Scipy

cheeting_factor = 0.9 # This is a subjective value to ease color bar scaling
distM = sp_dist.pdist(df, 'jaccard')
distM = sp_dist.squareform(distM)
np.fill_diagonal(distM, cheeting_factor)
distM = pd.DataFrame(distM, index=df.index, columns=df.index)

distM
```

```python
# Calculate the linkage for a dendrogram

linkage = sp_dendro.linkage(distM)
```

```python
# Show tissue source with colors

clust_colors = [colordict[habillage[x]] for x in df.index]
```

```python
# Visualize overlap on a heatmap

sns.clustermap(1-distM, row_colors=clust_colors, row_linkage=linkage, col_linkage=linkage)
```

At least we have the samples of the same pateint clustering together...  
On a separate dendrorgam (not sure if we should use the one from scanpy)

```python
# Draw the denrogram separately

fig, ax = plt.subplots()
sp_dendro.dendrogram(linkage, labels=df.index, ax=ax)
for lbl in ax.get_xticklabels():
    lbl.set_color(colordict[habillage[lbl.get_text()]])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_ticks([])
```

```python
# A Dendrogram based on gene expression - could be adapted?

sc.pl.dendrogram(adata, 'sample')
```
## Get some information on clonotypes typical for one of the groups


The rationale here would be to find clonotype clusters that occur preferentially for a diagnosis. For instance, clonotypes that are abundant or absent in tumor samples.

```python
# Define the cases we want to check

adata.obs['diagnosis'] = adata.obs['patient'].apply(lambda x: x[:-1])

clonotype_col = 'clonotype'
target_col = 'source'
target_label = 'Tumor'
control_label = None
replicate_col = 'sample'
additional_hue = 'diagnosis'
top_n = 10
```

```python
# Create a series of case-control groups for comparison

case_control_groups = []

group_cols = [target_col, replicate_col]
if additional_hue is None:
    hues = [None]
else:
    group_cols.append(additional_hue)
    hues = adata.obs[additional_hue].unique()
df = adata.obs.groupby(group_cols).agg('size').reset_index()

for hue in hues:
    if hue is None:
        tdf = df
    else:
        tdf = df.loc[df[additional_hue] == hue, :]
    cases = tdf.loc[df[target_col] == target_label,:]
    ncase = cases[0]
    cases = cases[replicate_col]
    if control_label is None:
        controls = tdf.loc[df[target_col] != target_label, :]
    else:
        controls = tdf.loc[df[target_col] == control_label, :]
    ncontrol = controls[0]
    controls = controls[replicate_col]
    case_control_groups.append([hue, cases, controls, ncase, ncontrol])
case_control_groups
```

```python
# Create a table of clonotype sizes (many steps identical to the table used for repertoire overlap)

clonotype_presence = adata.obs.loc[:, [clonotype_col, replicate_col]].copy()
clonotype_presence = clonotype_presence.groupby([clonotype_col, replicate_col]).agg('size').reset_index()
clonotype_presence = clonotype_presence.pivot(index=replicate_col, columns=clonotype_col, values=0)
clonotype_presence = clonotype_presence.fillna(0)

clonotype_presence
```

```python
# Do some statistics to compare between groupings for all clonotypes - Khi square and Fischer's came to my mind now, but any sugestion is welcome (also not sure if clonotype size or presence should be used)

from scipy.stats import fisher_exact

clt_stats, clt_selection, volcano = [], [], [[], []]
if control_label is None:
    control_label = 'Background'
for hue, cases, controls, ncase, ncontrol in case_control_groups:
    if hue is None:
        hue = 'All'
    tdf1 = clonotype_presence.loc[cases, ]
    tdf2 = clonotype_presence.loc[controls, ]
    suspects = set(tdf1.loc[:, tdf1.sum() > 0].columns.values.tolist() + tdf2.loc[:, tdf2.sum() > 0].columns.values.tolist())
    for suspect in suspects:
        case_sizes = tdf1[suspect]
        control_sizes = tdf2[suspect]
        rel_case_sizes = case_sizes/np.array(ncase)
        rel_control_sizes = control_sizes/np.array(ncontrol)
        np.mean((case_sizes+0.0001)/np.array(ncase))
        case_mean_freq = np.mean((case_sizes+0.0001)/np.array(ncase))
        case_presence = case_sizes.sum()
        case_absence = ncase.sum() - case_presence
        control_mean_freq = np.mean((control_sizes+0.0001)/np.array(ncontrol))
        control_presence = control_sizes.sum()
        control_absence = ncontrol.sum() - control_presence
        oddsratio, p = fisher_exact([[case_presence, control_presence], [case_absence, control_absence]])
        logfoldchange = np.log2(case_mean_freq/control_mean_freq)
        volcano[0].append(logfoldchange)
        volcano[1].append(-np.log10(p))
        clt_selection.append([p, suspect])
        for e in rel_case_sizes.index.values:
            clt_stats.append((suspect, hue, target_label, rel_case_sizes[e], e))
        for e in rel_control_sizes.index.values:
            clt_stats.append((suspect, hue, control_label, rel_control_sizes[e], e))
    
plt.scatter(*volcano, s=2, alpha=0.3)
```

```python
# Choose the most inbalanced clonotypes

clt_selection.sort(key=lambda x: x[0])
clt_selection = clt_selection[:top_n]
clt_selection
```

```python
# Draw clonotype size for the compared groups, accross hue - would be nice to tweak group abundances later, but now it seems a bit far

top_clonotypes = list(zip(*clt_selection))[1] 

if additional_hue is None:
    tclt_df = pd.DataFrame.from_records(clt_stats, columns =[clonotype_col, target_col, 'Normalized abundance', replicate_col])
    tclt_df = tclt_df.loc[tclt_df[clonotype_col].isin(top_clonotypes), :] 
    tclt_df = tclt_df.pivot_table(index=[target_col, replicate_col], columns=clonotype_col, values='Normalized abundance', fill_value=0).reset_index()
    tclt_df = pd.melt(tclt_df, id_vars=[target_col, replicate_col], value_name='Normalized abundance')
    sns.boxplot(x=clonotype_col, y='Normalized abundance', hue=target_col, data=tclt_df)
else:
    tclt_df = pd.DataFrame.from_records(clt_stats, columns =[clonotype_col, additional_hue, target_col, 'Normalized abundance', replicate_col])
    tclt_df = tclt_df.loc[tclt_df[clonotype_col].isin(top_clonotypes), :] 
    tclt_df = tclt_df.pivot_table(index=[additional_hue, target_col, replicate_col], columns=clonotype_col, values='Normalized abundance', fill_value=0).reset_index()
    tclt_df = pd.melt(tclt_df, id_vars=[additional_hue, target_col, replicate_col], value_name='Normalized abundance')
    sns.factorplot(x=clonotype_col, y='Normalized abundance', hue=target_col, kind='box', col=additional_hue, data=tclt_df)
```

```python
top_n = 20
```

```python
sns.factorplot(x=clonotype_col, y='Normalized abundance', hue=target_col, kind='bar', col=additional_hue, data=tclt_df)
```

```python
sns.catplot(x=clonotype_col, y='Normalized abundance', hue=target_col, kind='strip', col=additional_hue, data=tclt_df, dodge=True, col_wrap=2)
```

```python
# Try to show VDJ usage for one of the clonotypes only

ir.pl.vdj_usage(adata[adata.obs['clonotype'].isin(['2'])])
```

One single clonotype, as a result of convergence from different VDJ recombinations


## Comparison of gene expression in two clusters


If one finds a cluster of T cells with promising reactivity, it is a straightforward question if there is also a difference in terms of gene expression over other cells or another cluster.

```python
# Subset adata for cells that belong to either clonotype

adata.obs['grouping_for_gex'] = adata.obs['clonotype'].map({'2': 'Clonotype_2', '16': 'Clonotype_16'})
s_adata = adata[~adata.obs['grouping_for_gex'].isna()]
```

```python
# Have to do some prefiltering

adata.obs['grouping_for_gex'] = adata.obs['clonotype'].map({'2': 'Clonotype_2', '16': 'Clonotype_16'})
s_adata = adata[~adata.obs['grouping_for_gex'].isna()]
```

```python
# Check highest expressed genes

sc.pl.highest_expr_genes(s_adata, n_top=20, )
```

```python
# Compare gene expression between the two clonotypes

sc.tl.rank_genes_groups(s_adata, groupby='grouping_for_gex')
sc.pl.rank_genes_groups(s_adata, groupby='grouping_for_gex')
```

```python
# Show expression of top DE genes

deg_df = s_adata.uns['rank_genes_groups'].copy()
deg_df.pop('params')
deg_df = pd.DataFrame(deg_df)
top_deg = deg_df[0:5]['names']
top_deg = np.array([[x for x in y] for y in top_deg])
top_deg = top_deg.flatten()
sc.pl.stacked_violin(s_adata, top_deg, groupby='grouping_for_gex')
```

```python
# Show cell type marker genes

marker_genes = ['CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ',  'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1', 'FCGR3A']
sc.pl.stacked_violin(s_adata, marker_genes, groupby='grouping_for_gex')
```

```python
# Show genes related toT cell activation

activation_genes = ['IL2', 'IL2RA', 'CD28', 'CTLA4', 'LAT', 'STAT1', 'FOS', 'BRAF'] # This is only a guessing list, but might be worth digging up the literature
sc.pl.stacked_violin(s_adata, activation_genes, groupby='grouping_for_gex')
```

It might be more a scanpy related issue, not that much our problem, but instead of putting them beneath each other as stacked violins, it would be better to put them next to each other, like grouped strippolts.


## Putting subplots together

```python
# A last comment: we might want to make it easier to make our figures multipanel

fig, axs = plt.subplots(2, 2, figsize=(7, 11))
axs = axs.flatten()
tissues = ['Lung', 'Colon', 'Endo', 'Renal']
for i in range(len(tissues)):
    axs[i] = ir.pl.group_abundance(adata[adata.obs['patient'].str[:-1] == tissues[i]], groupby="clonotype", target_col="patient", max_cols=10, ax=axs[i])
```

```python

```





https://github.com/theislab/scanpy/issues/397

```python

```
