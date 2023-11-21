# adopted from pytorch scheduler
import torch

import math
from collections import Counter
from bisect import bisect_right


class NoAnnealingW():
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, init_weight, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_weights = [init_weight]
        self.last_epoch = last_epoch

    def get_weight(self, epoch):
        return self.base_weights

    def _get_closed_form_w(self, epoch):
        return self.base_weights


class Step_decay():
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, step_size, start, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.base_weights = [start]
        self.gamma = gamma
        self.last_epoch = last_epoch

    def get_weight(self, epoch):
        return self._get_closed_form_w(epoch)

    def _get_closed_form_w(self, epoch):
        return [base_weight * self.gamma ** (epoch // self.step_size)
                for base_weight in self.base_weights]


class LinearLR():
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.005    if epoch >= 4
        >>> scheduler = LinearLR(self.opt, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1,
                 verbose=False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError('Starting multiplicative factor expected to be between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters):
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]


class CosineAnnealingW():
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, init_weight, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_weights = [init_weight]
        self.last_epoch = last_epoch

    def get_weight(self, epoch):

        if epoch == 0:
            return self.base_weights
        elif (epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [base_w + (base_w - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_w in self.base_weights]
        return [(1 + math.cos(math.pi * epoch / self.T_max)) /
                (1 + math.cos(math.pi * (epoch - 1) / self.T_max)) *
                (base_w - self.eta_min) + self.eta_min
                for base_w in self.base_weights]

    def _get_closed_form_w(self, epoch):
        return [self.eta_min + (base_w - self.eta_min) *
                (1 + math.cos(math.pi * epoch / self.T_max)) / 2
                for base_w in self.base_weights]


class CosineAnnealingW2():

    def __init__(self, init_weight, T_max, eta_min=0, last_epoch=-1, verbose=False, flat_tail=True):
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_weights = [init_weight]
        self.last_epoch = last_epoch
        self.flat_tail = flat_tail

    def get_weight(self, epoch):

        if epoch == 0:
            return self.base_weights
        if self.flat_tail:
            return [self._check_min(w, epoch) for w in self._get_closed_form_w(epoch)]
        else:
            return self._get_closed_form_w(epoch)

    def _check_min(self, w, epoch):
        if epoch > self.T_max:
            return self.eta_min if w >= self.eta_min else w
        else:
            return w

    def _get_closed_form_w(self, epoch):
        return [self.eta_min + (base_w - self.eta_min) *
                (1 + math.cos(math.pi * epoch / self.T_max)) / 2
                for base_w in self.base_weights]


class MultiStepLR():
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, init_weight, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.base_weights = [init_weight]
        self.gamma = gamma
        # self.last_epoch = last_epoch

    def get_weight(self, epoch):
        if epoch not in self.milestones:
            return [base_w for base_w in self.base_weights]
        self.base_weights = [base_w * self.gamma ** self.milestones[epoch]
                             for base_w in self.base_weights]
        return self.base_weights

    def _get_closed_form_w(self, epoch):
        milestones = list(sorted(self.milestones.elements()))
        return [base_w * self.gamma ** bisect_right(milestones, epoch)
                for base_w in self.base_weights]


class StepLinearWarmpup():

    def __init__(self, init_value:list, T_max, start_step=0, eta_min=1.0, last_epoch=-1, verbose=False):
        self.T_max = (T_max-start_step)
        self.eta_min = eta_min
        self.base_weights = init_value
        self.last_epoch = last_epoch
        self.start_step = start_step

    def get_weight(self, epoch):

        if epoch <= self.start_step:
            return torch.Tensor(self.base_weights)
        elif epoch > self.T_max:
            return torch.Tensor([self.eta_min for base_w in self.base_weights])
        else:
            return self._get_closed_form_w(epoch - self.start_step)

    def _check_min(self, w, epoch):
        if epoch > self.T_max:
            return self.eta_min if w >= self.eta_min else w
        else:
            return w

    def _get_closed_form_w(self, epoch):
        return torch.Tensor([self.eta_min + (base_w - self.eta_min) *
                (1 + math.sin(math.pi * epoch / self.T_max + math.pi / 2)) / 2
                for base_w in self.base_weights])
