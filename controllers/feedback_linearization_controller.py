import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        M = self.model.M(x)
        C = self.model.C(x)
        q1, q2, q1_dot, q2_dot = x
        q = [q1, q2]
        q_dot = [q1_dot, q2_dot]

        Kd = 13
        Kp = 8

        ret = M @ (q_r_ddot - Kd * (q_dot - q_r_dot) - Kp * (q - q_r)) + C @ q_dot
        return ret
