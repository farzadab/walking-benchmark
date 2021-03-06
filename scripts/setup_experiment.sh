#!/bin/bash
################################################
#              setup experiments               #
#                                              #
#  make sure to that your current configs.yaml #
# is up to date before running this script     #
################################################
set -e

num_procs=10  # configs.yaml should not contain `num_processes` or `seed`
num_replicates=1
today=`date '+%Y_%m_%d__%H_%M_%S'`

name=$1
if [ $# -eq 0 ]
then
    echo "No arguments supplied: experiment name required"
    exit 1
fi

log_path=runs/${today}__${name}
mkdir -p runs
mkdir $log_path
cp configs.yaml $log_path/
cat > $log_path/run_script.sh <<EOF
#!/bin/bash
#SBATCH --account=rrg-vandepan
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=$num_procs
#SBATCH --mem-per-cpu=400M
#SBATCH --array=1-$num_replicates
module load singularity/3.2
cd `pwd`    # run this from the walking-benchmark folder
cd ..       # going to the parent dir, i.e. cproj
command="python -m main --log_dir $log_path/\$SLURM_ARRAY_TASK_ID \
    --seed \$SLURM_ARRAY_TASK_ID \
    --configs_path $log_path \
    --num_processes $num_procs --name $name--run\$SLURM_ARRAY_TASK_ID"
singularity exec -B /home -B /project -B /scratch imgwalking.sif bash -c "cd walking-benchmark && \$command"
EOF

cd $log_path

for ((i=1;i<=$num_replicates;i++)) do
    mkdir $i
done

sbatch run_script.sh
