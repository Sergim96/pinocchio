#
# Copyright (c) 2026 INRIA
#

import numpy as np
import pinocchio as pin


def compute_frame_placement_derivatives(
    model,
    data,
    q,
    v,
    frame_id,
    joint_placement_jacobians,
    reference_frame=pin.LOCAL,
    workspace=None,
):
    r"""Differentiate frame kinematics with respect to joint placements.

    This operation concerns one frame relative to the universe. It is not the
    derivative of a general two-joint ``RigidConstraintModel``. The topology,
    joint models and frame placement relative to its parent joint are fixed.

    Each non-universe joint placement is perturbed on the right according to

    .. math::

        P_i(p + \delta p)
        = P_i(p)\exp_6(K_i\delta p) + o(\|\delta p\|).

    ``joint_placement_jacobians`` stacks :math:`K_i` for joints
    ``1, ..., model.njoints - 1``. Pinocchio orders each tangent as three linear
    coordinates followed by three angular coordinates. A native-layout matrix
    containing an additional ignored universe block is also accepted, allowing
    the same seed matrix to be shared with ``computeRNEAPlacementDerivatives``.

    The analytical recursions are implemented in C++ and update ``data`` with
    the nominal second-order kinematics at zero generalized acceleration.

    Parameters
    ----------
    model : pinocchio.Model
        Rigid-body model whose joint placements are parameterized.
    data : pinocchio.Data
        Data associated with ``model``.
    q, v : array_like
        Configuration and generalized velocity.
    frame_id : int
        Frame whose kinematic quantities are differentiated.
    joint_placement_jacobians : array_like
        Stacked right-trivialized joint-placement Jacobians with shape
        ``(6 * (model.njoints - 1), np)`` or ``(6 * model.njoints, np)``.
    reference_frame : pinocchio.ReferenceFrame, optional
        ``LOCAL``, ``LOCAL_WORLD_ALIGNED`` or ``WORLD``.
    workspace : pinocchio.FramePlacementDerivativesWorkspace, optional
        Reusable native workspace. Supplying one avoids workspace allocation
        across repeated calls.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(d_placement_dp, dJ_dp, d_spatial_drift_dp,
        d_classical_drift_dp)`` with shapes ``(6, np)``,
        ``(6, model.nv, np)``, ``(6, np)`` and ``(6, np)``. Placement
        derivatives are right-trivialized in ``LOCAL`` coordinates regardless
        of ``reference_frame``. The spatial drift is the 6D :math:`\dot Jv`
        used by spatial constraints. The linear part of the classical drift is
        the :math:`\dot Jv` used by point constraints.
    """
    if not 0 <= frame_id < model.nframes:
        raise ValueError(f"frame_id must be in [0, {model.nframes})")
    if reference_frame not in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED, pin.WORLD):
        raise ValueError("reference_frame must be LOCAL, LOCAL_WORLD_ALIGNED or WORLD")

    dtype = data.J.dtype
    q = np.asarray(q, dtype=dtype)
    v = np.asarray(v, dtype=dtype)
    if q.shape != (model.nq,):
        raise ValueError(f"q must have shape ({model.nq},); got {q.shape}")
    if v.shape != (model.nv,):
        raise ValueError(f"v must have shape ({model.nv},); got {v.shape}")

    placement_jacobians = np.asarray(joint_placement_jacobians, dtype=dtype)
    valid_rows = (6 * (model.njoints - 1), 6 * model.njoints)
    if placement_jacobians.ndim != 2 or placement_jacobians.shape[0] not in valid_rows:
        raise ValueError(
            "joint_placement_jacobians must have shape "
            f"({valid_rows[0]}, np) or ({valid_rows[1]}, np); "
            f"got {placement_jacobians.shape}"
        )
    placement_jacobians = np.asfortranarray(placement_jacobians)

    if workspace is None:
        return pin.computeFramePlacementDerivatives(
            model,
            data,
            q,
            v,
            frame_id,
            placement_jacobians,
            reference_frame,
        )
    return pin.computeFramePlacementDerivatives(
        model,
        data,
        q,
        v,
        frame_id,
        placement_jacobians,
        reference_frame,
        workspace,
    )


def compute_frame_placement_derivatives_full(
    model,
    data,
    q,
    v,
    frame_id,
    reference_frame=pin.LOCAL,
    workspace=None,
):
    r"""Differentiate a frame with respect to every joint-placement direction.

    This convenience wrapper uses an identity direction matrix, giving six
    independent parameters for every non-universe joint. See
    :func:`compute_frame_placement_derivatives` for the general contracted
    operation.
    """
    number_parameters = 6 * (model.njoints - 1)
    return compute_frame_placement_derivatives(
        model,
        data,
        q,
        v,
        frame_id,
        np.eye(number_parameters, dtype=data.J.dtype),
        reference_frame,
        workspace,
    )
