#!/bin/bash
#SBATCH -o %jpro.out
#SBATCH -p cpu
#SBATCH -J PER
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --hint=multithread
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhuxk@act.buaa.edu.cn

srun hostname -s | sort -n > slurm.hosts
nohup python process_songs_.py >> log.log &
