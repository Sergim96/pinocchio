#
# Copyright (c) 2026 INRIA
#

import numpy as np
import pinocchio as pin


def compute_rigid_constraint_kinematic_derivatives(
    model,
    data,
    constraint_model,
    constraint_data,
    q,
    v,
    joint_placement_jacobians,
    workspace=None,
):
    r"""Differentiate a rigid two-frame constraint with respect to joint placements.

    Each non-universe joint placement is perturbed on the right by the seed
    matrix in ``joint_placement_jacobians``. Both moving constraint frames are
    evaluated from one nominal forward-kinematics and joint-Jacobian pass.

    Parameters
    ----------
    model, data : pinocchio.Model, pinocchio.Data
        Rigid-body model and its associated data.
    constraint_model, constraint_data
        A ``RigidConstraintModel`` and its associated data. Both ``CONTACT_3D``
        and ``CONTACT_6D``, in ``LOCAL`` or ``LOCAL_WORLD_ALIGNED``, are
        supported.
    q, v : array_like
        Configuration and generalized velocity.
    joint_placement_jacobians : array_like
        Stacked right-trivialized joint-placement seeds with shape
        ``(6 * (model.njoints - 1), np)`` or ``(6 * model.njoints, np)``.
    workspace : pinocchio.RigidConstraintKinematicDerivativesWorkspace, optional
        Reusable native workspace.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(d_relative_placement_dp, d_placement_error_dp, dJ_dp,
        d_velocity_error_dp, d_drift_dp)`` with shapes ``(6, np)``,
        ``(nc, np)``, ``(nc, model.nv, np)``, ``(nc, np)`` and ``(nc, np)``.
        Relative-placement tangents are right-trivialized in constraint frame
        2. Constraint-sized outputs use ``constraint_model.reference_frame``.
    """
    if constraint_model.type not in (
        pin.ContactType.CONTACT_3D,
        pin.ContactType.CONTACT_6D,
    ):
        raise ValueError("constraint_model.type must be CONTACT_3D or CONTACT_6D")
    if constraint_model.reference_frame not in (
        pin.LOCAL,
        pin.LOCAL_WORLD_ALIGNED,
    ):
        raise ValueError(
            "constraint_model.reference_frame must be LOCAL or LOCAL_WORLD_ALIGNED"
        )

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

    arguments = (
        model,
        data,
        constraint_model,
        constraint_data,
        q,
        v,
        placement_jacobians,
    )
    if workspace is None:
        return pin.computeRigidConstraintKinematicDerivatives(*arguments)
    return pin.computeRigidConstraintKinematicDerivatives(*arguments, workspace)


def compute_rigid_constraint_kinematic_derivatives_full(
    model,
    data,
    constraint_model,
    constraint_data,
    q,
    v,
    workspace=None,
):
    r"""Differentiate a rigid constraint along every joint-placement direction."""
    number_parameters = 6 * (model.njoints - 1)
    return compute_rigid_constraint_kinematic_derivatives(
        model,
        data,
        constraint_model,
        constraint_data,
        q,
        v,
        np.eye(number_parameters, dtype=data.J.dtype),
        workspace,
    )
