import torch
from torch import nn


class MultiCRVDO(nn.Module):
    def __init__(self, dt=torch.tensor(1 / 128), ):
        """
        Based on: Al Themairi, A.; Mahmoud, G.M.; Farghaly, A.A.; Abed-Elhameed, T.M. Complex Rayleigh–van-der-Pol–Duffing Oscillators: Dynamics, Phase, Antiphase Synchronization, and Image Encryption. Fractal Fract. 2023, 7, 886. https://doi.org/10.3390/fractalfract7120886
        MultiCRVDO - simulation of multiple CRVDO oscillators using Runge-Kutta 4(5) integration.

        :param dt: Time step for RK45
        """
        super().__init__()
        self.dt = nn.Buffer(dt)

    @staticmethod
    @torch.jit.script
    def crvdo_autonomous_batch(y, params, control):
        """
        Defines the system of differential equations for autonomous CRVDO (Complex Rayleigh–van-der-Pol–Duffing Oscillators)
        oscillators with different parameters in parallel.

        :param y: Current state tensor of shape [batch_size, 4], where 4 represents [z1, z2, z3, z4].
        :param params: Oscillator parameters tensor of shape [batch_size, 5], where 5 represents [a, b, c, d, e].
        :param control: Control function
        :return: Derivatives of the system's state, tensor of shape [batch_size, 4].
        """
        z1, z2, z3, z4 = y.unbind(-1)
        a, b, c, d, e = params.unbind(-1)
        u1, u2, u3, u4 = control.unbind(-1)

        return torch.stack([z3 + u1, z4,
                            z1 - a * (z1.pow(3) - 3 * z1 * z2.pow(2)) + e * (
                                    b * z3 - c * ((z1.pow(2) - z2.pow(2)) * z3 - 2 * z1 * z2 * z4) - d * (
                                    z3.pow(3) - 3 * z3 * z4.pow(2)) + u3),
                            z2 - a * (-z2.pow(3) + 3 * z1.pow(2) * z2) + e * (
                                    b * z4 - c * ((z1.pow(2) - z2.pow(2)) * z4 + 2 * z1 * z2 * z3) - d * (
                                    -z4.pow(3) + 3 * z2.pow(3) * z4))], dim=-1)

    def forward(self, y, params, control):
        """
        Forward function of MultiCRVDO.

        :param y: Current state tensor of shape [num_oscillators, 4], where 4 represents [z1, z2, z3, z4].
        :param params: Parameters of oscillators, tensor of shape [num_oscillators, 5], where 5 represents [a, b, c, d, e].
        :param control: Control function (input), tensor of shape [num_oscillators, 4].
        :return:
        """
        dt = self.dt

        # RK4 Integration
        k1 = self.crvdo_autonomous_batch(y, params, control)
        k2 = self.crvdo_autonomous_batch(y + 0.5 * dt * k1, params, control)
        k3 = self.crvdo_autonomous_batch(y + 0.5 * dt * k2, params, control)
        k4 = self.crvdo_autonomous_batch(y + dt * k3, params, control)

        return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_and_plot():
    model = MultiCRVDO()
    NUM = 10
    STEPS = 2000
    y0 = torch.tensor([[0.0, 0.0, 0.0, 0.0] for _ in range(NUM)])
    params = torch.tensor([[1.25, 2.0, 1 / (16 + i), 1.0, 0.25] for i in range(NUM)])
    control = torch.tensor([[1 / (2 + i), 0.0, 1 / (2 + i), 0.0] for i in range(NUM)])

    y = y0
    results = []
    for _ in range(STEPS):
        y = model.forward(y=y, params=params, control=control)
        results.append(y.clone())

    # Stack results and convert to numpy array for plotting
    results = torch.stack(results).detach().numpy()

    # Plot both oscillators' states over time
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12, 8))
    for j in range(NUM):
        for i in (0, 2):  # states 1 and 3 in autonomous CRVDO are always 0
            plt.plot(results[:, j, i], label=f'Oscillator {j + 1} State {i + 1}')
    plt.title('Oscillator States Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('States')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    simulate_and_plot()
