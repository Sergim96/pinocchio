#
# Copyright (c) 2026 INRIA
#

import numpy as np
import pinocchio as pin


def compute_rnea_placement_derivatives(
    model, data, q, v, a, joint_placement_jacobians, workspace=None
):
    r"""Compute RNEA derivatives with respect to joint placements.

    The kinematic topology, joint models and inertias are kept fixed. The input
    ``joint_placement_jacobians`` has shape ``(model.njoints, 6, np)``. Its
    block :math:`K_i` is the right-trivialized derivative of the placement
    :math:`P_i` stored in ``model.jointPlacements[i]``:

    .. math::

        P_i(p + \delta p)
        = P_i(p)\exp_6(K_i\delta p) + o(\|\delta p\|).

    The returned matrix has shape ``(model.nv, np)``. It is the kinematic
    contribution to :math:`\partial \tau / \partial p`; derivatives of the
    inertias themselves are not included.

    The computation is implemented by a fused native RNEA and derivative
    recursion. It updates ``data`` with the nominal RNEA quantities. No finite
    differences are used.

    Parameters
    ----------
    model : pinocchio.Model
        Rigid-body model whose joint placements are parameterized.
    data : pinocchio.Data
        Data associated with ``model``.
    q, v, a : array_like
        Configuration, velocity and acceleration passed to RNEA.
    joint_placement_jacobians : array_like
        Right-trivialized placement Jacobians, with shape
        ``(model.njoints, 6, np)``. The universe block at index zero is ignored.
    workspace : pinocchio.RNEAPlacementDerivativesWorkspace, optional
        Reusable native workspace. Supplying one avoids workspace allocation
        across repeated calls.

    Returns
    -------
    numpy.ndarray
        Generalized torque derivative with shape ``(model.nv, np)``.
    """
    placement_jacobians = np.asarray(joint_placement_jacobians)
    expected_prefix = (model.njoints, 6)
    if (
        placement_jacobians.ndim != 3
        or placement_jacobians.shape[:2] != expected_prefix
    ):
        raise ValueError(
            "joint_placement_jacobians must have shape "
            f"({model.njoints}, 6, np); got {placement_jacobians.shape}"
        )

    number_parameters = placement_jacobians.shape[2]
    stacked_jacobians = np.asfortranarray(
        placement_jacobians.reshape(model.njoints * 6, number_parameters),
        dtype=data.tau.dtype,
    )

    if workspace is None:
        return pin.computeRNEAPlacementDerivatives(
            model, data, q, v, a, stacked_jacobians
        )
    return pin.computeRNEAPlacementDerivatives(
        model, data, q, v, a, stacked_jacobians, workspace
    )
