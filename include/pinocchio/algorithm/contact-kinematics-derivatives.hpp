//
// Copyright (c) 2026 INRIA
//

#ifndef __pinocchio_algorithm_contact_kinematics_derivatives_hpp__
#define __pinocchio_algorithm_contact_kinematics_derivatives_hpp__

// IWYU pragma: begin_keep
#include "pinocchio/constraints.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"
// IWYU pragma: end_keep

namespace pinocchio
{
  ///
  /// \brief Reusable workspace for rigid-constraint kinematic derivatives with respect to joint
  /// placements.
  ///
  template<typename _Scalar, int _Options = 0>
  struct RigidConstraintKinematicDerivativesWorkspaceTpl
  {
    typedef _Scalar Scalar;
    enum
    {
      Options = _Options
    };

    typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic, Options> Matrix6x;
    typedef Eigen::Index Index;

    RigidConstraintKinematicDerivativesWorkspaceTpl()
    : frame_workspace()
    , frame_parameter_storage(6, 0)
    , frame_jacobian_derivative_storage(6, 0)
    , frame_jacobian_storage(6, 0)
    , nv(0)
    , parameter_capacity(0)
    {
    }

    RigidConstraintKinematicDerivativesWorkspaceTpl(const Index nv, const Index parameter_capacity)
    : frame_workspace()
    , frame_parameter_storage(6, 0)
    , frame_jacobian_derivative_storage(6, 0)
    , frame_jacobian_storage(6, 0)
    , nv(0)
    , parameter_capacity(0)
    {
      resize(nv, parameter_capacity);
    }

    void resize(const Index new_nv, const Index new_parameter_capacity)
    {
      assert(new_nv >= 0);
      assert(new_parameter_capacity >= 0);
      nv = new_nv;
      parameter_capacity = new_parameter_capacity;
      frame_workspace.resize(nv, parameter_capacity);
      frame_parameter_storage.resize(6, 6 * parameter_capacity);
      frame_jacobian_derivative_storage.resize(6, 2 * nv * parameter_capacity);
      frame_jacobian_storage.resize(6, 3 * nv);
    }

    bool isCompatible(const Index expected_nv, const Index number_parameters) const
    {
      return nv == expected_nv && parameter_capacity >= number_parameters;
    }

    auto framePlacementDerivative(const Index frame_id, const Index number_parameters)
    {
      return parameterBlock(frame_id, number_parameters);
    }

    auto frameAccelerationDerivative(const Index frame_id, const Index number_parameters)
    {
      return parameterBlock(2 + frame_id, number_parameters);
    }

    auto frameClassicalAccelerationDerivative(const Index frame_id, const Index number_parameters)
    {
      return parameterBlock(4 + frame_id, number_parameters);
    }

    auto frameJacobianDerivative(const Index frame_id, const Index number_parameters)
    {
      assert(frame_id >= 0 && frame_id < 2);
      return frame_jacobian_derivative_storage.middleCols(
        frame_id * nv * parameter_capacity, nv * number_parameters);
    }

    auto frameJacobian(const Index frame_id)
    {
      assert(frame_id >= 0 && frame_id < 2);
      return frame_jacobian_storage.middleCols(frame_id * nv, nv);
    }

    auto localConstraintJacobian()
    {
      return frame_jacobian_storage.middleCols(2 * nv, nv);
    }

    FramePlacementDerivativesWorkspaceTpl<Scalar, Options> frame_workspace;
    Matrix6x frame_parameter_storage;
    Matrix6x frame_jacobian_derivative_storage;
    Matrix6x frame_jacobian_storage;
    Index nv;
    Index parameter_capacity;

  protected:
    auto parameterBlock(const Index block_id, const Index number_parameters)
    {
      assert(block_id >= 0 && block_id < 6);
      assert(number_parameters >= 0 && number_parameters <= parameter_capacity);
      return frame_parameter_storage.middleCols(block_id * parameter_capacity, number_parameters);
    }
  };

  typedef RigidConstraintKinematicDerivativesWorkspaceTpl<context::Scalar, context::Options>
    RigidConstraintKinematicDerivativesWorkspace;

  ///
  /// \brief Computes the derivatives of a two-frame rigid constraint with respect to
  /// right-trivialized joint-placement perturbations.
  ///
  /// A single forward-kinematics and joint-Jacobian pass is shared by the two constraint frames.
  /// The Jacobian derivative is flattened parameter-major: middleCols(p * model.nv, model.nv)
  /// contains the derivative for parameter p. Relative-placement derivatives are always
  /// right-trivialized in frame 2. All constraint-sized outputs follow cmodel.reference_frame.
  /// For LOCAL_WORLD_ALIGNED, placement_error_partial_dp is the rotated placement sensitivity
  /// used by the Baumgarte stabilization term. constraint_acceleration_partial_dp differentiates
  /// the unconstrained kinematic drift \f$\dot{J}v\f$ (spatial for CONTACT_6D, classical linear
  /// for CONTACT_3D).
  ///
  /// \param[in] model The rigid-body model.
  /// \param[in,out] data Data updated with nominal zero-generalized-acceleration kinematics.
  /// \param[in] cmodel The two-frame rigid constraint model.
  /// \param[in,out] cdata Constraint data updated with the two nominal frame placements.
  /// \param[in] q Joint configuration vector (dim model.nq).
  /// \param[in] v Joint velocity vector (dim model.nv).
  /// \param[in] joint_placement_jacobians Stacked placement seeds with 6 * model.njoints or
  /// 6 * (model.njoints - 1) rows and number_parameters columns.
  /// \param[in,out] workspace Reusable workspace compatible with model.nv and the parameter count.
  /// \param[out] relative_placement_partial_dp Right-trivialized c1Mc2 derivative expressed in
  /// constraint frame 2 (dim 6 by number_parameters).
  /// \param[out] placement_error_partial_dp Constraint placement-error derivative (dim nc by
  /// number_parameters).
  /// \param[out] constraint_jacobian_partial_dp Parameter-major flattened Jacobian derivative
  /// (dim nc by model.nv * number_parameters).
  /// \param[out] constraint_velocity_partial_dp Constraint velocity-error derivative (dim nc by
  /// number_parameters).
  /// \param[out] constraint_acceleration_partial_dp Constraint drift derivative (dim nc by
  /// number_parameters).
  ///
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType,
    typename PlacementJacobianType,
    typename RelativePlacementDerivativeType,
    typename PlacementErrorDerivativeType,
    typename JacobianDerivativeType,
    typename VelocityDerivativeType,
    typename AccelerationDerivativeType>
  void computeRigidConstraintKinematicDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const RigidConstraintModelTpl<Scalar, Options> & cmodel,
    RigidConstraintDataTpl<Scalar, Options> & cdata,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType> & v,
    const Eigen::MatrixBase<PlacementJacobianType> & joint_placement_jacobians,
    RigidConstraintKinematicDerivativesWorkspaceTpl<Scalar, Options> & workspace,
    const Eigen::MatrixBase<RelativePlacementDerivativeType> & relative_placement_partial_dp,
    const Eigen::MatrixBase<PlacementErrorDerivativeType> & placement_error_partial_dp,
    const Eigen::MatrixBase<JacobianDerivativeType> & constraint_jacobian_partial_dp,
    const Eigen::MatrixBase<VelocityDerivativeType> & constraint_velocity_partial_dp,
    const Eigen::MatrixBase<AccelerationDerivativeType> & constraint_acceleration_partial_dp);

} // namespace pinocchio

// IWYU pragma: begin_exports
#include "pinocchio/src/algorithm/contact-kinematics-derivatives.hxx"
// IWYU pragma: end_exports

#endif // __pinocchio_algorithm_contact_kinematics_derivatives_hpp__
