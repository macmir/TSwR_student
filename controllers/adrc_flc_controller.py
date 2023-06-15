import numpy as np
# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = None
        self.Kp = Kp
        self.Kd = Kd
        p1 = p[0]
        p2 = p[1]
        self.model = ManiuplatorModel(Tp)
        self.L = np.array([[3*p1, 0], [0, 3*p2], [3*p1**2, 0], [0, 3*p2**2], [p1**3, 0], [0, p2**3]])
        W = np.array([[1.0, 0.0, 0.0, 0.0, 0.0 , 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        A = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        B = np.zeros((6, 2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        A = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # print(A)

        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        M = np.linalg.inv(M)
        C = self.model.C(x)
        MC = -(M @ C)
        print(A.shape)
        A[2, 2] = MC[0, 0]
        A[2, 3] = MC[0, 1]
        A[3, 2] = MC[1, 0]
        A[3, 3] = MC[1, 1]

        B = np.zeros((6, 2))

        B[2, 0] = MC[0, 0]
        B[2, 1] = MC[0, 1]
        B[3, 0] = MC[1, 0]
        B[3, 1] = MC[1, 1]
        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        M = self.model.M(x)
        C = self.model.C(x)

        z_hat = self.eso.get_state()
        x_hat = z_hat[0:2]
        x_hat_dot = z_hat[2:4]

        f = z_hat[4:]

        e = q_d - q
        e_dot = q_d_dot - x_hat_dot

        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e
        u = M @ (v - f) + C @ x_hat_dot
        self.update_params(x_hat, x_hat_dot)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1))

        return u