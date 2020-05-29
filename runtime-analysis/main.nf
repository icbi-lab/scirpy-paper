#!/usr/bin/env nextflow

process simulate_tcr {
    publishDir "results/01_simulate", mode: params.publishDirMode

    conda = "/home/sturm/anaconda3/envs/immunesim"

    input:
        val chain from Channel.from(['a', 'b'])

    output: 
        file "cdr3_${chain}.tsv"

    """
    simulate_cdr3.R $chain 500000 cdr3_${chain}.tsv
    """
}
