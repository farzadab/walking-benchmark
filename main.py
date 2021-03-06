import os
import glob
import copy
import time
import random
import gym
import warnings
import multiprocessing
import pybullet_envs
import numpy as np
import torch as th
from torch.optim.lr_scheduler import ExponentialLR
from colorama import Fore, Style

from utils.argparser import ArgsFromFile
from utils.logs import LogMaster
from utils.envs import auto_tune_env, get_mirror_function, SymEnv
from utils.normalization import NormalizedEnv
from utils import infrange


from machina.algos import ppo_clip
from machina.traj import Traj
from machina.samplers import EpiSampler
from machina.traj import epi_functional as ef
from machina.utils import measure, set_device
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv
from machina import logger

from simple_net import PolNet, PolNetB, VNet, VNetB, PolNetLSTM, VNetLSTM
from symmetric_net import SymmetricNet, SymmetricValue, SymmetricStats
from sym_net import SymNet, SymVNet


import mocca_envs
import pybullet_envs
import roboschool
import envs


# TODO: this mirroring is becoming a pain point, gotta refactor


class Trainer(object):
    default_args = {
        "net_version": 2,
        "mirror_tuples": False,
        "lr_decay_gamma": 0.992,
        "plot": True,
        "evaluate": False,
        "eval_epis": 50,
        "mirror": False,
    }

    def __init__(self, args=None):
        self.env = None
        self.sampler = None
        self.experiment = None
        self.c_level = None
        self.pol = None

        if args is None:
            args = ArgsFromFile(self.default_args)
        self.args = args

        assert args.mirror in [False, "old", "new", "tuple"]

        if args.mirror == "old":
            args.mirror = True

        args.mirror_tuples = args.mirror == "tuple"

        self.env_kwargs = copy.deepcopy(getattr(self.args, "env_kwargs", {}))

        device_name = "cpu" if args.cuda < 0 else "cuda:{}".format(args.cuda)
        set_device(th.device(device_name))

        self.num_updates = (
            int(args.num_total_frames) // args.num_steps // args.num_processes
        )

        self.seed_torch(args.seed)

        self.setup_env()

        if args.render or args.evaluate:
            self.load_save(args.env, "last")
        else:
            if hasattr(args, "load_path"):
                self.load_save(args.env, "last")
            else:
                self.setup_nets()
            self.setup_optims()
            self.setup_experiment()

        # if args.cuda:
        #     self.nets.cuda()

    def seed_torch(self, seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        if self.args.cuda:
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            th.backends.cudnn.benchmark = False
            th.backends.cudnn.deterministic = True

    def setup_env(self):
        """
            :params ratio: should be a float between 0 and 1
        """
        env_id = self.args.env
        if self.env_kwargs:
            env_id = auto_tune_env(env_id, self.env_kwargs)
        env = GymEnv(
            env_id,
            log_dir=os.path.join(self.args.load_path, "movie")
            if self.args.render
            else None,
            record_video=self.args.record,
        )
        env.env.seed(self.args.seed)
        if self.args.c2d:
            env = C2DEnv(env)

        if self.env is None:
            self.env = NormalizedEnv(env)
            if self.args.mirror is True:
                if hasattr(env.unwrapped, "mirror_sizes"):
                    self.env.stats = SymmetricStats(
                        *env.unwrapped.mirror_sizes[:3], max_obs=4000
                    )
                else:
                    self.args.mirror = False
            elif self.args.mirror == "new":
                self.env = SymEnv(self.env)
        else:
            # don't want to override the normalization
            self.env.replace_wrapped_env(env)

    def get_curriculum_level(self, ratio):
        # Digitize the ratio into N levels
        return (
            np.digitize(
                ratio, np.linspace(0, 1 + 1e-6, self.args.curriculum["levels"] + 1)
            )
            - 1
        ) / (self.args.curriculum["levels"] - 1)

    def curriculum_handler(self, ratio):
        if not hasattr(self.args, "curriculum"):
            return False

        if "levels" not in self.args.curriculum:
            raise ValueError(
                "Curriculum levels not specified: `configs.yaml` shoud have an integer `curriculum.levels`"
            )

        c_level = self.get_curriculum_level(ratio)

        print("Curriculum Level:", c_level)

        if self.c_level != c_level:
            print("Leveled up!")
            self.c_level = c_level

            self.handle_env_curr(self.c_level)
            self.handle_stdev_curr(self.c_level)

            return True
        return False

    def handle_env_curr(self, c_level):
        curric = self.args.curriculum
        if "env_kwargs" in curric:
            curric_kwargs = dict(
                [
                    # TODO: enable more than two end-points for the linear interpolation
                    (k, v[0] * (1 - c_level) + v[1] * c_level)
                    for k, v in curric["env_kwargs"].items()
                ]
            )
            if (
                getattr(curric, "compensate_power", False)
                and "power_coef" in curric_kwargs
            ):
                curric_kwargs["action_coef"] = (
                    curric["env_kwargs"]["power_coef"][0] / curric_kwargs["power_coef"]
                )
            self.env_kwargs.update(curric_kwargs)
            self.setup_env()

    def handle_stdev_curr(self, c_level):
        if "log_stdev" in self.args.curriculum and self.pol is not None:
            start, finish = self.args.curriculum["log_stdev"]
            self.pol.net.reset_log_std(
                finish * self.c_level + start * (1 - self.c_level)
            )

    def setup_nets(self):
        ob_space = self.env.observation_space
        ac_space = self.env.action_space

        if self.args.mirror is True:
            print("Initiating a symmetric network")
            pol_net = SymmetricNet(
                *self.env.unwrapped.mirror_sizes,
                hidden_size=int(self.args.hidden_size / 4),
                num_layers=self.args.num_layers,
                varying_std=self.args.varying_std,
                tanh_finish=self.args.tanh_finish,
                log_std=self.args.log_stdev,
            )
        elif self.args.rnn:
            pol_net = PolNetLSTM(ob_space, ac_space, h_size=256, cell_size=256)
        elif self.args.net_version == 1:
            pol_net = PolNet(ob_space, ac_space, log_std=self.args.log_stdev)
        else:
            pol_net = PolNetB(
                ob_space,
                ac_space,
                hidden_size=self.args.hidden_size,
                num_layers=self.args.num_layers,
                varying_std=self.args.varying_std,
                tanh_finish=self.args.tanh_finish,
                log_std=self.args.log_stdev,
            )

        if self.args.mirror == "new":
            print("Initiating a new symmetric network")
            # TODO: in this case the action_space for the previous pol_net is incorrect, but it isn't easy to fix ...
            # we can use this for now which just ignores some of the final indices
            pol_net = SymNet(
                pol_net,
                ob_space.shape[0],
                *self.env.unwrapped.sym_act_inds,
                varying_std=self.args.varying_std,
                log_std=self.args.log_stdev,
                deterministic=False,
            )

        if isinstance(ac_space, gym.spaces.Box):
            pol_class = GaussianPol
        elif isinstance(ac_space, gym.spaces.Discrete):
            pol_class = CategoricalPol
        elif isinstance(ac_space, gym.spaces.MultiDiscrete):
            pol_class = MultiCategoricalPol
        else:
            raise ValueError("Only Box, Discrete, and MultiDiscrete are supported")

        policy = pol_class(
            ob_space,
            ac_space,
            pol_net,
            self.args.rnn,
            data_parallel=self.args.data_parallel,
            parallel_dim=1 if self.args.rnn else 0,
        )

        if self.args.mirror is True:
            vf_net = SymmetricValue(
                *self.env.unwrapped.mirror_sizes[:3],
                hidden_size=self.args.hidden_size,
                num_layers=self.args.num_layers,
            )
        elif self.args.rnn:
            vf_net = VNetLSTM(ob_space, h_size=256, cell_size=256)
        elif self.args.net_version == 1:
            vf_net = VNet(ob_space)
        else:
            vf_net = VNetB(
                ob_space,
                hidden_size=self.args.hidden_size,
                num_layers=self.args.num_layers,
            )

        if self.args.mirror == "new":
            print("Initiating a new symmetric value network")
            vf_net = SymVNet(vf_net, ob_space.shape[0])

        vf = DeterministicSVfunc(
            ob_space,
            vf_net,
            self.args.rnn,
            data_parallel=self.args.data_parallel,
            parallel_dim=1 if self.args.rnn else 0,
        )

        self.pol = policy
        self.vf = vf

    def setup_optims(self):
        self.optim_pol = th.optim.Adam(self.pol.net.parameters(), self.args.pol_lr)
        self.optim_vf = th.optim.Adam(self.vf.net.parameters(), self.args.vf_lr)
        self.scheduler_pol = ExponentialLR(self.optim_pol, self.args.lr_decay_gamma)
        self.scheduler_vf = ExponentialLR(self.optim_vf, self.args.lr_decay_gamma)

    def setup_experiment(self):
        self.logger = LogMaster(self.args)
        self.logger.store_exp_data({})
        self.args.store_configs(self.logger.log_dir)
        self.writer = self.logger.get_writer()

    def train(self):
        args = self.args

        # TODO: cuda seems to be broken, I don't care about it right now
        # if args.cuda:
        #     # current_obs = current_obs.cuda()
        #     rollouts.cuda()

        self.train_start_time = time.time()
        total_epi = 0
        total_step = 0
        max_rew = -1e6
        sampler = None

        score_file = os.path.join(self.logger.get_logdir(), "progress.csv")
        logger.add_tabular_output(score_file)

        num_total_frames = args.num_total_frames

        mirror_function = None
        if args.mirror_tuples and hasattr(self.env.unwrapped, "mirror_indices"):
            mirror_function = get_mirror_function(**self.env.unwrapped.mirror_indices)
            num_total_frames *= 2
            if not args.tanh_finish:
                warnings.warn(
                    "When `mirror_tuples` is `True`,"
                    " `tanh_finish` should be set to `True` as well."
                    " Otherwise there is a chance of the training blowing up."
                )

        while num_total_frames > total_step:
            # setup the correct curriculum learning environment/parameters
            new_curriculum = self.curriculum_handler(total_step / args.num_total_frames)

            if total_step == 0 or new_curriculum:
                if sampler is not None:
                    del sampler
                sampler = EpiSampler(
                    self.env,
                    self.pol,
                    num_parallel=self.args.num_processes,
                    seed=self.args.seed + total_step,  # TODO: better fix?
                )

            with measure("sample"):
                epis = sampler.sample(
                    self.pol, max_steps=args.num_steps * args.num_processes
                )

            with measure("train"):
                with measure("epis"):
                    traj = Traj()
                    traj.add_epis(epis)

                    traj = ef.compute_vs(traj, self.vf)
                    traj = ef.compute_rets(traj, args.decay_gamma)
                    traj = ef.compute_advs(traj, args.decay_gamma, args.gae_lambda)
                    traj = ef.centerize_advs(traj)
                    traj = ef.compute_h_masks(traj)
                    traj.register_epis()

                    if mirror_function:
                        traj.add_traj(mirror_function(traj))

                # if args.data_parallel:
                #     self.pol.dp_run = True
                #     self.vf.dp_run = True

                result_dict = ppo_clip.train(
                    traj=traj,
                    pol=self.pol,
                    vf=self.vf,
                    clip_param=args.clip_eps,
                    optim_pol=self.optim_pol,
                    optim_vf=self.optim_vf,
                    epoch=args.epoch_per_iter,
                    batch_size=args.batch_size if not args.rnn else args.rnn_batch_size,
                    max_grad_norm=args.max_grad_norm,
                )

                # if args.data_parallel:
                #     self.pol.dp_run = False
                #     self.vf.dp_run = False

            ## append the metrics to the `results_dict` (reported in the progress.csv)
            result_dict.update(self.get_extra_metrics(epis))

            total_epi += traj.num_epi
            step = traj.num_step
            total_step += step
            rewards = [np.sum(epi["rews"]) for epi in epis]
            mean_rew = np.mean(rewards)
            logger.record_results(
                self.logger.get_logdir(),
                result_dict,
                score_file,
                total_epi,
                step,
                total_step,
                rewards,
                plot_title=args.env,
            )

            if mean_rew > max_rew:
                self.save_models("max")
                max_rew = mean_rew

            self.save_models("last")

            self.scheduler_pol.step()
            self.scheduler_vf.step()

            del traj

    def get_model_names(self):
        return [
            "pol",
            "vf",
            # 'optim_pol', 'optim_vf',
            "env__state",
        ]

    def load_save(self, env_name, name=None):

        load_path = os.path.join(self.args.load_path, "models")

        ext = (("_" + str(name)) if name else "") + ".pt"

        for mname in self.get_model_names():
            # TODO: ugly, need to be prettier
            try:
                state_dict = th.load(os.path.join(load_path, mname + ext))
            except:
                warnings.warn('Could not load model "%s"' % mname)
                continue
            if "__state" in mname:
                getattr(self, mname.split("__")[0]).load_state_dict(state_dict)
            else:
                setattr(self, mname, state_dict)

        self.env.disable_update()

    def get_extra_metrics(self, epis):
        reported_keys = epis[0]["e_is"].keys()
        metrics = {}
        for k in reported_keys:
            metrics["Mean" + k] = np.mean([m for epi in epis for m in epi["e_is"][k]])
            if "rew" in k.lower():
                epi_values = [np.sum(epi["e_is"][k]) for epi in epis]
                metrics["MeanEpi" + k] = np.mean(epi_values)
                metrics["MaxEpi" + k] = np.max(epi_values)

        return metrics

    def save_models(self, name=None):
        save_path = os.path.join(self.logger.get_logdir(), "models")
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        ext = (("_" + str(name)) if name else "") + ".pt"

        for mname in self.get_model_names():
            model = getattr(self, mname.split("__")[0])
            if hasattr(model, "cpu"):
                model = model.cpu()
            if "__state" in mname:
                model = model.state_dict()
            fpath = os.path.join(save_path, mname + ext)
            th.save(model, fpath)

    def evaluate(self):
        # use the env at the end of the curriculum
        self.curriculum_handler(1)

        print("current log stds:", self.pol.net.log_std_param)
        env = self.env

        bullet = "Bullet" in self.args.env
        if bullet and self.args.render:
            env.render()  # mode='human')

        #    if "Roboschool" in self.args.env:
        #        from OpenGL import GLU

        rews = {}

        for _ in range(self.args.eval_epis):
            obs = env.reset()
            sum_rew = {"Rew": 0}

            for _ in infrange():
                action = self.pol.deterministic_ac_real(th.FloatTensor(obs))[0].reshape(
                    -1
                )
                obs, reward, done, info = env.step(action)
                sum_rew["Rew"] += reward  # info["ProgressRew"]

                for k, v in info.items():
                    if "rew" in k.lower():
                        sum_rew[k] = sum_rew.get(k, 0) + v

                if self.args.render:
                    env.render()
                if done:
                    break

            print("return: ", sum_rew)
            for k, v in sum_rew.items():
                rews[k] = rews.get(k, []) + [v]

        with open(os.path.join(self.args.load_path, "evaluate.csv"), "w") as csvfile:
            csvfile.write(",".join(["MeanEpi" + k for k in rews.keys()]) + "\n")
            csvfile.write(",".join([str(np.mean(rews[k])) for k in rews.keys()]) + "\n")


def main():
    trainer = Trainer()
    if trainer.args.render or trainer.args.evaluate:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
