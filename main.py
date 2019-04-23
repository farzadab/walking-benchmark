import os
import glob
import copy
import time
import gym
import warnings
import multiprocessing
import pybullet_envs
import numpy as np
import torch as th
from colorama import Fore, Style

from utils.argparser import ArgsFromFile
from utils.logs import LogMaster
from utils.envs import auto_tune_env
from utils.normalization import NormalizedEnv


from machina.algos import ppo_clip
from machina.traj import Traj
from machina.samplers import EpiSampler
from machina.traj import epi_functional as ef
from machina.utils import measure, set_device
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv
from machina import logger

from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM


import cassie_pybullet_env.envs
import pybullet_envs
import roboschool


class Trainer(object):
    config_filename = 'configs.yaml'
    def __init__(self, args=None):
        if args is None:
            args = ArgsFromFile(self.config_filename)
        self.args = args

        device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
        set_device(th.device(device_name))

        self.num_updates = int(args.num_total_frames) // args.num_steps // args.num_processes
        self.experiment = None

        th.set_num_threads(args.num_processes)

        th.manual_seed(args.seed)
        if args.cuda:
            th.cuda.manual_seed(args.seed)

        self.setup_env()

        if args.render:
            self.load_save(args.env, 'last')
        else:
            self.setup_experiment()
            self.setup_nets()

        # if args.cuda:
        #     self.nets.cuda()
    
    def setup_env(self):
        if hasattr(self.args, 'env_kwargs') and 'Roboschool' in self.args.env:
            env_id = auto_tune_env(self.args.env, self.args.env_kwargs)
        else:
            env_id = self.args.env
        env = GymEnv(
            env_id,
            log_dir=os.path.join(self.args.log_dir, 'movie'),
            record_video=self.args.record,
        )
        env.env.seed(self.args.seed)
        if self.args.c2d:
            env = C2DEnv(env)
        self.env = NormalizedEnv(env)
    
    def setup_nets(self):
        ob_space = self.env.observation_space
        ac_space = self.env.action_space

        if self.args.rnn:
            pol_net = PolNetLSTM(ob_space, ac_space, h_size=256, cell_size=256)
        else:
            pol_net = PolNet(ob_space, ac_space)

        if isinstance(ac_space, gym.spaces.Box):
            pol_class = GaussianPol
        elif isinstance(ac_space, gym.spaces.Discrete):
            pol_class = CategoricalPol
        elif isinstance(ac_space, gym.spaces.MultiDiscrete):
            pol_class = MultiCategoricalPol
        else:
            raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')
        
        policy = pol_class(ob_space, ac_space, pol_net, self.args.rnn,
                           data_parallel=self.args.data_parallel,
                           parallel_dim=1 if self.args.rnn else 0)

        if self.args.rnn:
            vf_net = VNetLSTM(ob_space, h_size=256, cell_size=256)
        else:
            vf_net = VNet(ob_space)

        vf = DeterministicSVfunc(ob_space, vf_net, self.args.rnn,
                                 data_parallel=self.args.data_parallel, parallel_dim=1 if self.args.rnn else 0)
        
        self.pol = policy
        self.vf = vf

        self.optim_pol = th.optim.Adam(pol_net.parameters(), self.args.pol_lr)
        self.optim_vf = th.optim.Adam(vf_net.parameters(), self.args.vf_lr)
    

    def setup_experiment(self):
        self.logger = LogMaster(self.args)
        self.logger.store_exp_data({})
        self.args.store_configs(self.logger.log_dir)
        self.writer = self.logger.get_writer()


    def train(self):
        args = self.args

        sampler = EpiSampler(self.env, self.pol, num_parallel=self.args.num_processes, seed=self.args.seed)

        # TODO: cuda seems to be broken, I don't care about it right now
        # if args.cuda:
        #     # current_obs = current_obs.cuda()
        #     rollouts.cuda()

        self.train_start_time = time.time()
        total_epi = 0
        total_step = 0
        max_rew = -1e6

        score_file = os.path.join(self.logger.get_logdir(), 'progress.csv')
        logger.add_tabular_output(score_file)


        while args.num_total_frames > total_step:
            with measure('sample'):
                epis = sampler.sample(self.pol, max_steps=args.num_steps * args.num_processes)
            with measure('train'):
                traj = Traj()
                traj.add_epis(epis)

                traj = ef.compute_vs(traj, self.vf)
                traj = ef.compute_rets(traj, args.decay_gamma)
                traj = ef.compute_advs(traj, args.decay_gamma, args.gae_lambda)
                traj = ef.centerize_advs(traj)
                traj = ef.compute_h_masks(traj)
                traj.register_epis()

                # if args.data_parallel:
                #     self.pol.dp_run = True
                #     vf.dp_run = True

                result_dict = ppo_clip.train(traj=traj, pol=self.pol, vf=self.vf, clip_param=args.clip_eps,
                                             optim_pol=self.optim_pol, optim_vf=self.optim_vf,
                                             epoch=args.epoch_per_iter,
                                             batch_size=args.batch_size if not args.rnn else args.rnn_batch_size,
                                             max_grad_norm=args.max_grad_norm)

                # if args.data_parallel:
                #     self.pol.dp_run = False
                #     vf.dp_run = False


            ## append the metrics to the `results_dict` (reported in the progress.csv)
            result_dict.update(self.get_extra_metrics(epis))

            total_epi += traj.num_epi
            step = traj.num_step
            total_step += step
            rewards = [np.sum(epi['rews']) for epi in epis]
            mean_rew = np.mean(rewards)
            logger.record_results(self.logger.get_logdir(), result_dict, score_file,
                                  total_epi, step, total_step,
                                  rewards,
                                  plot_title=args.env)

            if mean_rew > max_rew:
                self.save_models('max')
                max_rew = mean_rew

            self.save_models('last')
            
            del traj
        del sampler
    
    def get_model_names(self):
        return [
            'pol', 'vf',
            # 'optim_pol', 'optim_vf',
            'env__state',
        ]

    def load_save(self, env_name, name=None):

        load_path = os.path.join(self.args.load_path, 'models')

        ext = (('_' + str(name)) if name else '') + '.pt'

        for mname in self.get_model_names():
            # TODO: ugly, need to be prettier
            try:
                state_dict = th.load(os.path.join(load_path, mname + ext))
            except:
                warnings.warn('Could not load model "%s"' % mname)
            if '__state' in mname:
                getattr(self, mname.split('__')[0]).load_state_dict(state_dict)
            else:
                setattr(self, mname, state_dict)
        
        self.env.disable_update()


    def get_extra_metrics(self, epis):
        reported_keys = epis[0]['e_is'].keys()
        metrics = {}
        for k in reported_keys:
            metrics['Mean' + k] = np.mean([m for epi in epis for m in epi['e_is'][k]])
            if 'rew' in k.lower():
                metrics['MeanEpi' + k] = np.mean([np.sum(epi['e_is'][k]) for epi in epis])

        return metrics
        

    def save_models(self, name=None):
        save_path = os.path.join(self.logger.get_logdir(), 'models')
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        
        ext = (('_' + str(name)) if name else '') + '.pt'

        for mname in self.get_model_names():
            model = getattr(self, mname.split('__')[0])
            if hasattr(model, 'cpu'):
                model = model.cpu()
            if '__state' in mname:
                model = model.state_dict()
            fpath = os.path.join(save_path, mname + ext)
            th.save(model, fpath)


    def render(self):        
        env = self.env
        done = True

        bullet = 'Bullet' in self.args.env
        if bullet:
            env.render()#mode='human')
        
        if 'Roboschool' in self.args.env:
            from OpenGL import GLU
        

        total_reward = 0

        # import ipdb
        # ipdb.set_trace()

        while True:
            if done:
                print(total_reward)
                obs = env.reset()
                total_reward = 0
            
            if bullet:
                # TODO fix this hack: the pybullet code is broken :|
                env.unwrapped._p.resetDebugVisualizerCamera(5, 0, -20, env.unwrapped.robot.body_xyz)
                # env.unwrapped.body_xyz = env.unwrapped.robot.body_xyz
                # env.unwrapped.camera._p = env.unwrapped._p
                # env.unwrapped.camera_adjust()#distance=5, yaw=0)
            else:
                env.render()#mode='human')

            action = self.pol.deterministic_ac_real(th.FloatTensor(obs))[0].reshape(-1)
            # print(action.shape)
            # print(action)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            time.sleep(1/60)


def main():
    trainer = Trainer()
    if trainer.args.render:
        trainer.render()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
