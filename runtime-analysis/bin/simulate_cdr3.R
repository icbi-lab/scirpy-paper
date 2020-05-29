#!/usr/bin/env Rscript

##################################################
# USAGE: 
#  simulate_cdr3.R [a,b] n output_file.tsv
# 
# where a = alpha chains, b = beta chains
# n = number of seqs
##################################################

library(readr)
library(immuneSIM)
args = commandArgs(trailingOnly=TRUE)

chain = args[1]
n = as.numeric(args[2])
out_file = args[3]

sim_repertoire <- immuneSIM(
     number_of_seqs = n,
     species = "hs",
     receptor = "tr",
     chain = chain,
     user_defined_alpha=2,
     verbose= TRUE)

write_tsv(sim_repertoire, out_file)
