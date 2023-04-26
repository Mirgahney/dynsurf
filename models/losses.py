import copy
import json
import math
import os
import pathlib
from typing import Any, Callable, List, Optional, Text, Tuple, Union

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import collections

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any
Activation = Callable[[Array], Array]
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]
Normalizer = Callable[[], Callable[[Array], Array]]


def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
  """Auxiliary function of radial_and_tangential_undistort()."""

  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[np.ndarray, np.ndarray]:
  """Computes undistorted (x, y) from (xd, yd)."""
  # Initialize from the distorted point.
  x = xd.copy()
  y = yd.copy()

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = np.where(
        np.abs(denominator) > eps, x_numerator / denominator,
        np.zeros_like(denominator))
    step_y = np.where(
        np.abs(denominator) > eps, y_numerator / denominator,
        np.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y


def log1p_safe(x):
  """The same as tf.math.log1p(x), but clamps the input to prevent NaNs."""
  return torch.log1p(torch.minimum(x, 3e37*torch.ones_like(x)))


def exp_safe(x):
  """The same as tf.math.exp(x), but clamps the input to prevent NaNs."""
  return torch.exp(torch.minimum(x, 87.5*torch.ones_like(x)))


def expm1_safe(x):
  """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
  return torch.expm1(torch.minimum(x, 87.5*torch.ones_like(x)))


def safe_sqrt(x, eps=1e-7):
  safe_x = torch.where(x == 0, torch.ones_like(x) * eps, x)
  return torch.sqrt(safe_x)


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size

    based on code from https://github.com/ast0414/adversarial-example/blob/master/craft.py
    """
    assert inputs.requires_grad

    num_classes = output.size()[-1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[..., i] = 1
        output.backward(grad_output, retain_graph=True) #retain_variables=True)
        jacobian[i] = inputs.grad.data

    return jacobian.permute(1, 2, 0, 3)


def general_loss_with_squared_residual(squared_x, alpha, scale):
  r"""The general loss that takes a squared residual.
  This fuses the sqrt operation done to compute many residuals while preserving
  the square in the loss formulation.
  This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.
  Args:
    squared_x: The residual for which the loss is being computed. x can have
      any shape, and alpha and scale will be broadcasted to match x's shape if
      necessary.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
      interpolation between several discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha.
  Returns:
    The losses for each element of x, in the same shape as x.

    adpted from: https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/utils.py#L1
  """
  eps = torch.finfo(torch.float32).eps

  # This will be used repeatedly.
  squared_scaled_x = squared_x / (scale ** 2)

  # The loss when alpha == 2.
  loss_two = 0.5 * squared_scaled_x
  # The loss when alpha == 0.
  loss_zero = log1p_safe(0.5 * squared_scaled_x)
  # The loss when alpha == -infinity.
  loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
  # The loss when alpha == +infinity.
  loss_posinf = expm1_safe(0.5 * squared_scaled_x)

  # The loss when not in one of the above special cases.
  # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
  beta_safe = np.maximum(eps, np.abs(alpha - 2.))
  # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
  alpha_safe = np.where(
      np.greater_equal(alpha, 0.), np.ones_like(alpha),
      -np.ones_like(alpha)) * np.maximum(eps, np.abs(alpha))
  loss_otherwise = (beta_safe / alpha_safe) * (
      torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

  # Select which of the cases of the loss to return.
  alpha = torch.tensor(alpha)
  loss = torch.where(
      alpha == -torch.inf, loss_neginf,
      torch.where(
          alpha == 0, loss_zero,
          torch.where(
              alpha == 2, loss_two,
              torch.where(alpha == torch.inf, loss_posinf, loss_otherwise))))

  return scale * loss


def compute_elastic_loss(jacobian, eps=1e-6, loss_type='log_svals'):
  """Compute the elastic regularization loss.
  The loss is given by sum(log(S)^2). This penalizes the singular values
  when they deviate from the identity since log(1) = 0.0,
  where D is the diagonal matrix containing the singular values.
  Args:
    jacobian: the Jacobian of the point transformation.
    eps: a small value to prevent taking the log of zero.
    loss_type: which elastic loss type to use.
  Returns:
    The elastic regularization loss.
    adpted from: https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/training.py
  """
  if loss_type == 'log_svals':
    svals = torch.linalg.svdvals(jacobian)
    log_svals = torch.log(torch.maximum(svals, eps*torch.ones_like(svals)))
    sq_residual = torch.sum(log_svals**2, dim=-1)
  elif loss_type == 'svals':
    svals = torch.linalg.svdvals(jacobian)
    sq_residual = torch.sum((svals - 1.0)**2, dim=-1)
  elif loss_type == 'jtj':
    jtj = jacobian @ jacobian.T
    sq_residual = ((jtj - torch.eye(3)) ** 2).sum() / 4.0
  #elif loss_type == 'div':
  #  div = utils.jacobian_to_div(jacobian)
  #  sq_residual = div ** 2
  elif loss_type == 'det':
    det = torch.linalg.det(jacobian)
    sq_residual = (det - 1.0) ** 2
  elif loss_type == 'log_det':
    det = torch.linalg.det(jacobian)
    sq_residual = torch.log(torch.maximum(det, eps)) ** 2
  #elif loss_type == 'nr':
  #  rot = nearest_rotation_svd(jacobian)
  #  sq_residual = torch.sum((jacobian - rot) ** 2)
  else:
    raise NotImplementedError(
        f'Unknown elastic loss type {loss_type!r}')
  residual = torch.sqrt(sq_residual)
  loss = general_loss_with_squared_residual(
      sq_residual, alpha=-2.0, scale=0.03)
  return loss, residual


def compute_akap_loss(jacobian, eps=1e-6, loss_type='log_svals'):
  """Compute the As killing As Possible regularization loss.
  The loss is given by sum(log(S)^2). This penalizes the singular values
  when they deviate from the identity since log(1) = 0.0,
  where D is the diagonal matrix containing the singular values.
  Args:
    jacobian: the Jacobian of the point transformation.
    eps: a small value to prevent taking the log of zero.
    loss_type: which elastic loss type to use.
  Returns:
    The elastic regularization loss.
    adpted from: https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/training.py
  """

  energy = jacobian + jacobian.transpose(1, 2)
  akap_loss = 0.5 * torch.linalg.norm(energy, dim=(1, 2))
  loss = general_loss_with_squared_residual(
      akap_loss, alpha=-2.0, scale=0.03)
  return loss


def alternating_charge(r, h, q0: float):

    phi = r
    theta = np.pi/(2.*(2.*np.pi - h)) * (phi - 2.*np.pi) + np.pi
    delta = r >= h
    return delta * q0 * torch.cos(theta)


def compute_alternating_electric_field_loss(qp, volume_origin, volume_dim, h, q0: float, lamda_per=0.0795):
    """
    Args:
        qp:
        volume_origin:
        volume_dim:
        h:
        q0:
        lamda_per: 1/4πε
            water:87.9, 80.2, 55.5 (0, 20, 100 °C), air: 1.0006 and vacuum: 8.854 x 10-12
    Returns:

    """
    qp_norm = 2. * (qp - volume_origin[None, None, :]) / volume_dim[None, None, :] - 1.
    q_qp = alternating_charge(qp_norm, h, q0)
    F = lamda_per * q0 * q_qp
    return -F.mean()


# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    if det.nonzero().shape[0] == 0:
        print(f"det {det}")
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


def scale_aux_loss(scale, lambda_1=0.1, lambda_2=0.1):
    return (lambda_1 * torch.maximum(torch.zeros_like(scale), -scale) + lambda_2 * torch.maximum(scale - 1., torch.zeros_like(scale))).mean()


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based', lambda_1=0.5, lambda_2=0.5):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha
        self.__scale_loss = lambda scale: scale_aux_loss(scale, lambda_1=lambda_1, lambda_2=lambda_2)

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        scale_loss = self.__scale_loss(scale)

        return total, scale_loss

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


def get_smooth_loss(model_outputs):
    # smoothness loss as unisurf
    g1 = model_outputs['grad_theta']
    g2 = model_outputs['grad_theta_nei']

    normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
    normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
    smooth_loss = torch.norm(normals_1 - normals_2, dim=-1).mean()
    return smooth_loss


# normal consistency loss
def get_normal_loss(normal_pred, normal_gt):
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1).mean()
    cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
    return l1, cos