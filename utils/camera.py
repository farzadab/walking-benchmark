import numpy as np


class CameraPos(object):
    """
    A spring-damper model for the movement of a camera following a target
    """

    kp = 1
    kd = 1

    def __init__(self, pos, dt, offset):
        self.pos = np.array(pos)
        self.dt = dt
        self.offset = np.array(offset)
        self.vel = np.zeros(3)

    def followTarget(self, target):
        p_err = self.offset + target - self.pos
        v_err = -self.vel
        u = self.kp * p_err + self.kd * v_err
        self.pos += self.vel * self.dt
        self.vel += u * self.dt
        return self.pos
