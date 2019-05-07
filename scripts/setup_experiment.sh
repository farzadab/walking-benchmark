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
source $HOME/cproj/venv/bin/activate
cd `pwd`
python -m main --log_dir $log_path/\$SLURM_ARRAY_TASK_ID \
    --seed \$SLURM_ARRAY_TASK_ID \
    --load_path $log_path \
    --num_processes $num_procs --name $name--run\$SLURM_ARRAY_TASK_ID
EOF

cd $log_path

for ((i=1;i<=$num_replicates;i++)) do
    mkdir $i
done

sbatch run_script.sh
