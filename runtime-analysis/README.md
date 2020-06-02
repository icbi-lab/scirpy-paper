# Runtime analysis

## Prerequisites:
 * Nextflow
 * Setting up the conda environments `immunesim.yml` and `scirpy_runtime_benchmark.yml`. Immunesim is not avilable as conda package and needs to be installed manually. 
 * Jupyter notebook

## Running the analysis: 
1) simulate TCR sequences and run `scirpy.tcr_dist.tcr_dist` function with `benchmark_tcr_dist.nf`
2) simulate Anndata objects with clonotype sizes following a powerlaw distribution using `analyses/01_simulate-data.ipynb`
3) Run the benchmark of `scirpy.tl.tcr_neighbors` with `benchmark_tcr_neighbors.nf`
4) Generate the figures with `analyses/02_runtime-analysis.ipynb`. 
