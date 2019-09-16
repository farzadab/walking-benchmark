# PPO For Locomotion and Curriculum Learning
This repository contains an implementation of the [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) algorithm that I used for my research that was partly presented in my [MSc thesis](https://www.cs.ubc.ca/~farzadab/static/msc_thesis.pdf) (Chapter 4 - Torque Limit Considerations).

My research was supervised by [Michiel van de Panne](https://www.cs.ubc.ca/~van/) in the [Motion Capture and Character Animation](https://github.com/UBCMOCCA/) lab working on locomotion and reinforcement learning.

### Related Repositories
 - [SymmetricRL](https://github.com/UBCMOCCA/SymmetricRL/): focuses on incorporating symmetry into the RL paradigm
 - [mocca_envs](https://github.com/UBCMOCCA/mocca_envs/): a set of locomotion environments 

## Installation

There is no need for compilation. You can install all requirements using Pip, however, you might prefer to install some manully, including:
 - [PyTorch](https://pytorch.org/get-started/locally/)
 - [PyBullet](https://pybullet.org)


### Installation using Pip
```bash
# TODO: create and activate your virtual env of choice

# clone the repo
git clone https://github.com/farzadab/walking-benchmark

cd SymmetricRL
pip install -r requirements  # you might prefer to install some packages (including PyTorch) yourself
```


## Running Locally

To run an experiment named `test_experiment` with the PyBullet humanoid environment you can run:

```bash
./scripts/local_run_playground_train.sh  test_experiment
```

The `test_experiment` is the name of the experiment. This command will create a new experiment directory inside the `runs` directory that contains the following files:

- `slurm.out`: the output of the process. You can use `tail -f` to view the contents
- `configs.yaml`: a YAML file containing all the hyper-parameter values used in this run
- `1/pid`: the process ID of the task running the training algorithm
- `1/progress.csv`: a CSV file containing the data about the the training progress
- `1/variant.json`: extra useful stuff about the git commit (only works if `pygit2` is installed)
- `1/git_diff.patch`: git diff with the current commit (can be used along with variant to get to the exact code that was run)
- `1/models`: a directory containing the saved models

## Plotting Results
```bash
python -m scripts.plot_from_csv --load_path runs/*/*/  --columns RewardAverage RewardMax --name_regex '.*__([^\/]*)\/'  --smooth 2
```
It reads the `progress.csv` file inside each directory to plot the training curves.

## Running Learned Policy
```bash
python -m playground.enjoy with experiment_dir=runs/<EXPERIMENT_DIRECTORY>
```
Note that the `<EXPERIMENT_DIRECTORY>` does include the experiment number, i.e., `EXPERIMENT_DIRECTORY=runs/2019_09_06__14_23_08__test_experiment/1/`.

## Evaluating Results
```bash
python -m playground.evaluate with render=True experiment_dir=runs/<EXPERIMENT_DIRECTORY>
```
Results are outputed as a CSV file in the same directory under the name `evaluate.csv`.
