import math
import torch as th
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F

from simple_net import VNetB
from utils.normalization import Stats


class SymmetricLayer(nn.Module):
    __constants__ = ["sbias", "cbias"]
    __weights__ = ["c2s", "n2s", "c2c", "s2c", "s1s", "s2s", "n2n", "s2n"]

    def __init__(self, in_const, in_neg, in_side, out_const, out_neg, out_side, wmag=1):
        """
        """
        super().__init__()

        self.in_const = in_const
        self.in_neg = in_neg
        self.in_side = in_side
        self.out_const = out_const
        self.out_neg = out_neg
        self.out_side = out_side

        self.c2s = Parameter(th.Tensor(out_side, in_const))
        self.s1s = Parameter(th.Tensor(out_side, in_side))
        self.s2s = Parameter(th.Tensor(out_side, in_side))
        self.n2s = Parameter(th.Tensor(out_side, in_neg))
        self.sbias = Parameter(th.Tensor(out_side))

        self.s2c = Parameter(th.Tensor(out_const, in_side))
        self.c2c = Parameter(th.Tensor(out_const, in_const))
        self.cbias = Parameter(th.Tensor(out_const))

        self.n2n = Parameter(th.Tensor(out_neg, in_neg))
        self.s2n = Parameter(th.Tensor(out_neg, in_side))

        self.reset_parameters(wmag)

    def forward(self, c, n, l, r):
        c2s = F.linear(c, self.c2s, self.sbias)
        n2s = F.linear(n, self.n2s)
        s2c = F.linear((l + r) / 2, self.s2c, self.cbias)
        s2n = F.linear(l - r, self.s2n)
        return (
            s2c + F.linear(c, self.c2c),  # constant part
            s2n + F.linear(n, self.n2n),  # negative part
            c2s + n2s + F.linear(l, self.s1s) + F.linear(r, self.s2s),  # left
            c2s - n2s + F.linear(r, self.s1s) + F.linear(l, self.s2s),  # right
        )

    #         self.in_features = in_features
    #         self.out_features = out_features
    #         self.weight = Parameter(th.Tensor(out_features, in_features))

    #         if bias:
    #             self.bias = Parameter(th.Tensor(out_features))
    #         else:
    #             self.register_parameter('bias', None)

    def reset_parameters(self, wmag):
        for wname in self.__weights__:
            init.kaiming_uniform_(getattr(self, wname), a=math.sqrt(5) * wmag)

        bound = 1 / math.sqrt(self.in_const + self.in_neg + 2 * self.in_side)
        self.sbias.data.fill_(0)
        self.cbias.data.fill_(0)
        # init.uniform_(self.sbias, -bound, bound)
        # init.uniform_(self.cbias, -bound, bound)


#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#     @weak_script_method
#     def forward(self, input):
#         return F.linear(input, self.weight, self.bias)

#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )


class SymmetricNet(nn.Module):
    def __init__(
        self,
        c_in,
        n_in,
        s_in,
        c_out,
        n_out,
        s_out,
        num_layers=3,
        hidden_size=16,
        tanh_finish=True,
        varying_std=False,
        log_std=-1,
        deterministic=False,
    ):
        super().__init__()
        self.c_in = c_in
        self.s_in = s_in
        self.n_in = n_in
        self.tanh_finish = tanh_finish
        self.deterministic = deterministic

        self.layers = []
        last_cin = c_in
        last_nin = n_in
        last_sin = s_in
        for i in range(num_layers - 1):
            self.layers.append(
                SymmetricLayer(
                    last_cin, last_nin, last_sin, hidden_size, hidden_size, hidden_size
                )
            )
            self.add_module("layer%d" % i, self.layers[i])
            last_cin, last_nin, last_sin = hidden_size, hidden_size, hidden_size
        self.layers.append(
            SymmetricLayer(last_cin, last_nin, last_sin, c_out, n_out, s_out, wmag=3e-3)
        )
        self.add_module("final", self.layers[-1])

        if not self.deterministic:
            action_size = c_out + n_out + 2 * s_out
            if varying_std:
                self.log_std_param = nn.Parameter(
                    th.randn(action_size) * 1e-10 + log_std
                )
            else:
                self.log_std_param = log_std * th.ones(action_size)

    @property
    def input_size(self):
        return self.c_in + self.n_in + 2 * self.s_in

    def forward(self, obs):
        cs, ns, ss = self.c_in, self.n_in, self.s_in
        c = obs.index_select(-1, th.arange(0, cs))
        n = obs.index_select(-1, th.arange(cs, cs + ns))
        l = obs.index_select(-1, th.arange(cs + ns, cs + ns + ss))
        r = obs.index_select(-1, th.arange(cs + ns + ss, cs + ns + 2 * ss))

        for i, layer in enumerate(self.layers):
            if i != 0:
                n = th.tanh(n)  # TODO
                c = th.relu(c)
                r = th.relu(r)
                l = th.relu(l)

            c, n, l, r = layer(c, n, l, r)

        mean = th.cat([c, n, l, r], -1)
        if self.tanh_finish:
            mean = th.tanh(mean)

        if not self.deterministic:
            log_std = self.log_std_param.expand_as(mean)
            return mean, log_std
        else:
            return mean


class SymmetricStats(Stats):
    def __init__(self, c_in, n_in, s_in, *args, **kwargs):
        self.zeros_inds = th.arange(c_in, c_in + n_in)
        self.shared_inds = th.stack(
            [
                th.arange(c_in + n_in, c_in + n_in + s_in),
                th.arange(c_in + n_in + s_in, c_in + n_in + 2 * s_in),
            ]
        )

    def observe(self, obs):
        """
        @brief update observation mean & stdev
        # @param obs: the observation. assuming NxS where N is the batch-size and S is the input-size
        @param obs: the observation (can be 1D or 2D in which case the first dimension is the batch-size)
        """
        if self.n > self.max_obs:
            return
        super().observe(obs)
        self.mean[self.zeros_inds] = 0

        shared_mean = self.mean[self.shared_inds].mean(0)
        shared_std = self.std[self.shared_inds].max(0)[0]

        for inds in self.shared_inds:
            self.mean[inds] = shared_mean
            self.std[inds] = shared_std


class SymmetricValue(nn.Module):
    def __init__(self, c_in, n_in, s_in, num_layers=3, hidden_size=64):
        super().__init__()
        self.c_in = c_in
        self.s_in = s_in
        self.n_in = n_in

        obs_space = th.zeros([c_in + n_in + 2 * s_in])

        self.net = VNetB(obs_space, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, obs):
        cs, ns, ss = self.c_in, self.n_in, self.s_in
        c = obs.index_select(-1, th.arange(0, cs))
        n = obs.index_select(-1, th.arange(cs, cs + ns))
        l = obs.index_select(-1, th.arange(cs + ns, cs + ns + ss))
        r = obs.index_select(-1, th.arange(cs + ns + ss, cs + ns + 2 * ss))

        return (self.net(obs) + self.net(th.cat([c, -n, r, l], -1))) / 2

