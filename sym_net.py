import torch as th
import torch.nn as nn
from machina.utils import measure


class SymNet(nn.Module):
    def __init__(
        self,
        net,
        in_size,
        c_out,
        n_out,
        s_out,
        log_std=-1,
        varying_std=False,
        deterministic=True,
    ):
        assert in_size % 2 == 0
        half = in_size / 2

        super().__init__()
        self.net = net
        self.deterministic = deterministic

        self.mirror_inds = th.cat([th.arange(half, in_size), th.arange(0, half)]).long()

        self.c_range = th.arange(0, c_out)
        self.n_range = th.arange(c_out, c_out + n_out)
        self.s_range = th.arange(c_out + n_out, c_out + n_out + s_out)

        if not self.deterministic:
            action_size = c_out + n_out + 2 * s_out
            if varying_std:
                self.log_std_param = nn.Parameter(
                    th.randn(action_size) * 1e-10 + log_std
                )
            else:
                self.log_std_param = log_std * th.ones(action_size)

    def forward(self, obs):
        mobs = obs.index_select(-1, self.mirror_inds)
        out_o = self.net(obs)
        out_m = self.net(mobs)
        if not self.deterministic:
            out_o = out_o[0]
            out_m = out_m[0]
        # original
        c_o = out_o.index_select(-1, self.c_range)
        n_o = out_o.index_select(-1, self.n_range)
        s_o = out_o.index_select(-1, self.s_range)
        # mirrored
        c_m = out_m.index_select(-1, self.c_range)
        n_m = out_m.index_select(-1, self.n_range)
        s_m = out_m.index_select(-1, self.s_range)

        mean = th.cat(
            [
                # commons
                c_o + c_m,
                # opposites/negatives
                n_o - n_m,
                # side 1
                s_o,
                # side 2
                s_m,
            ],
            -1,
        )

        if self.deterministic:
            return mean
        else:
            return mean, self.log_std_param.expand_as(mean)


class SymVNet(nn.Module):
    def __init__(self, net, in_size):
        assert in_size % 2 == 0
        half = in_size / 2
        super().__init__()
        self.net = net
        self.mirror_inds = th.cat([th.arange(half, in_size), th.arange(0, half)]).long()

    def forward(self, obs):
        mobs = obs.index_select(-1, self.mirror_inds)
        return self.net(obs) + self.net(mobs)

