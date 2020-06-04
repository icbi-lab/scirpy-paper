#!/bin/bash
#$ -cwd
#$ -S /bin/bash
#$ -V
#$ -R y
#$ -pe smp 4
#$ -q all.q@apollo-0*
jupytext --to ipynb analysis-wu2020.md
time jupyter nbconvert \
  --execute \
  --to html \
  --output-dir /home/sturm/scratch \
  --output wu2020.html \
  --ExecutePreprocessor.timeout=-1 \
  analysis-wu2020.ipynb
