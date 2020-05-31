#!/usr/bin/env nextflow

process benchmark_tcr_neighbors {
    publishDir "results/04_tcr_neighbors", mode: params.publishDirMode

    conda = "/home/sturm/.conda/envs/scirpy_runtime_benchmark"
    cpus = 1

    input:
        file adata from Channel.fromPath("results/03_simulate_adata/adata_*.h5ad")

    output:
        file "timing_${adata.baseName}.txt"
        file "timing_mem_${adata.baseName}.txt"

    """
    /usr/bin/time -v benchmark_tcr_neighbors.py ${adata} > timing_${adata.baseName}.txt 2> timing_mem_${adata.baseName}.txt
    """

}
