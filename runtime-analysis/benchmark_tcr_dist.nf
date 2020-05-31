#!/usr/bin/env nextflow

process simulate_tcr {
    publishDir "results/01_simulate", mode: params.publishDirMode

    conda = "/home/sturm/.conda/envs/immunesim"

    input:
        val chain from Channel.from(['a', 'b'])

    output:
        file "cdr3_${chain}.tsv" into simulated_cdr3

    """
    simulate_cdr3.R $chain 500000 cdr3_${chain}.tsv
    """
}

process benchmark_tcr_dist {
    publishDir "results/02_tcr_dist", mode: params.publishDirMode

    conda = "/home/sturm/.conda/envs/scirpy_runtime_benchmark"
    cpus = 16

    input:
        val n from Channel.from(["5000", "10000", "20000", "50000", "100000", "200000", "300000"])
        file tra_seqs from simulated_cdr3.collect()

    output:
        file "timing_${n}.txt"
        file "timing_mem_${n}.txt"

    """
    /usr/bin/time -v benchmark_tcr_dist.py ${tra_seqs[0]} $n ${task.cpus} > timing_${n}.txt 2> timing_mem_${n}.txt
    """

}
