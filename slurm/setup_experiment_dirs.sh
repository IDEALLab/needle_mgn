#!/bin/bash
# Create all experiment output directories on the HPC host before submitting jobs.
# Run this once on the login node before sbatch-ing the experiment scripts.

BASE=/home/nhoffma1/scratch.fuge-prj/needle_mgn/experiments

EXPERIMENTS=(
    domino_base
    domino_noise
    domino_fourier
    domino_cpress
    cropped_base
    cropped_noise
    cropped_fourier
    cropped_cpress
    cropped_stride1
    cropped_splitnorm
    cropped_large
    cropped_nocrop
    cropped_kan
    cropped_bistride
    cropped_downsampled
    cropped_fiber
    cropped_fiber_kan
    cropped_tfn
)

for exp in "${EXPERIMENTS[@]}"; do
    mkdir -p ${BASE}/${exp}/checkpoints
    mkdir -p ${BASE}/${exp}/stats
    mkdir -p ${BASE}/${exp}/outputs
    echo "Created ${BASE}/${exp}/{checkpoints,stats,outputs}"
done

echo ""
echo "All experiment directories created under ${BASE}"
echo "Submit jobs with: for f in slurm/*.sh; do sbatch \$f; done"
