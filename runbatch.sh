#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --time=01:00:00
#SBATCH --account=boempac
#SBATCH --job-name=osw_20_ext
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=patrick.duffy@nrel.gov

module load conda
conda activate /shared-projects/rev/modulefiles/conda/envs/rev/

# The directory where the job was submitted from
cd /home/pduffy/great_lakes/osw20/

# Run python script
python3 get_resource_one_site_parallel.py
