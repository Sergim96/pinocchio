//
// Copyright (c) 2026 INRIA
//

#pragma once

#include "pinocchio/spatial/explog.hpp"

namespace pinocchio
{
  namespace impl
  {
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
      const Eigen::MatrixBase<AccelerationDerivativeType> & constraint_acceleration_partial_dp)
    {
      typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
      typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;
      typedef typename Model::JointIndex JointIndex;
      typedef typename Data::Motion Motion;
      typedef typename Data::SE3 SE3;
      typedef typename Data::Matrix6 Matrix6;
      typedef typename Data::Vector3 Vector3;

      PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model.nq);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model.nv);
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        cmodel.type == CONTACT_3D || cmodel.type == CONTACT_6D,
        "The rigid constraint type must be CONTACT_3D or CONTACT_6D");
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        cmodel.reference_frame == LOCAL || cmodel.reference_frame == LOCAL_WORLD_ALIGNED,
        "The rigid constraint reference frame must be LOCAL or LOCAL_WORLD_ALIGNED");
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        cmodel.joint1_id < (JointIndex)model.njoints
          && cmodel.joint2_id < (JointIndex)model.njoints,
        "The rigid constraint joint indexes are not valid");
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        joint_placement_jacobians.rows() == 6 * model.njoints
          || joint_placement_jacobians.rows() == 6 * (model.njoints - 1),
        "The placement Jacobian must have 6 * model.njoints or 6 * (model.njoints - 1) rows");

      const Eigen::Index number_parameters = joint_placement_jacobians.cols();
      const Eigen::Index constraint_size = cmodel.residualSize();
      PINOCCHIO_CHECK_ARGUMENT_SIZE(relative_placement_partial_dp.rows(), 6);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(relative_placement_partial_dp.cols(), number_parameters);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(placement_error_partial_dp.rows(), constraint_size);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(placement_error_partial_dp.cols(), number_parameters);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(constraint_jacobian_partial_dp.rows(), constraint_size);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(
        constraint_jacobian_partial_dp.cols(), model.nv * number_parameters);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(constraint_velocity_partial_dp.rows(), constraint_size);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(constraint_velocity_partial_dp.cols(), number_parameters);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(constraint_acceleration_partial_dp.rows(), constraint_size);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(constraint_acceleration_partial_dp.cols(), number_parameters);
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        workspace.isCompatible(model.nv, number_parameters),
        "The rigid-constraint kinematic-derivative workspace is not compatible with model.nv or "
        "the parameter count");
      assert(model.check(data) && "data is not consistent with model.");

      RelativePlacementDerivativeType & relative_placement_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(RelativePlacementDerivativeType, relative_placement_partial_dp);
      PlacementErrorDerivativeType & placement_error_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(PlacementErrorDerivativeType, placement_error_partial_dp);
      JacobianDerivativeType & constraint_jacobian_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(JacobianDerivativeType, constraint_jacobian_partial_dp);
      VelocityDerivativeType & constraint_velocity_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(VelocityDerivativeType, constraint_velocity_partial_dp);
      AccelerationDerivativeType & constraint_acceleration_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(AccelerationDerivativeType, constraint_acceleration_partial_dp);

      pinocchio::forwardKinematics(model, data, q, v, workspace.frame_workspace.zero_acceleration);
      pinocchio::computeJointJacobians(model, data);
      cmodel.calc(model, data, cdata);

      auto frame1_placement_partial_dp = workspace.framePlacementDerivative(0, number_parameters);
      auto frame2_placement_partial_dp = workspace.framePlacementDerivative(1, number_parameters);
      auto frame1_jacobian_partial_dp = workspace.frameJacobianDerivative(0, number_parameters);
      auto frame2_jacobian_partial_dp = workspace.frameJacobianDerivative(1, number_parameters);
      auto frame1_acceleration_partial_dp =
        workspace.frameAccelerationDerivative(0, number_parameters);
      auto frame2_acceleration_partial_dp =
        workspace.frameAccelerationDerivative(1, number_parameters);
      auto frame1_classical_acceleration_partial_dp =
        workspace.frameClassicalAccelerationDerivative(0, number_parameters);
      auto frame2_classical_acceleration_partial_dp =
        workspace.frameClassicalAccelerationDerivative(1, number_parameters);

      computeFramePlacementDerivativesPrepared(
        model, data, cmodel.joint1_id, cmodel.joint1_placement, cdata.oMc1,
        joint_placement_jacobians, LOCAL, workspace.frame_workspace, frame1_placement_partial_dp,
        frame1_jacobian_partial_dp, frame1_acceleration_partial_dp,
        frame1_classical_acceleration_partial_dp);
      computeFramePlacementDerivativesPrepared(
        model, data, cmodel.joint2_id, cmodel.joint2_placement, cdata.oMc2,
        joint_placement_jacobians, LOCAL, workspace.frame_workspace, frame2_placement_partial_dp,
        frame2_jacobian_partial_dp, frame2_acceleration_partial_dp,
        frame2_classical_acceleration_partial_dp);

      auto frame1_jacobian = workspace.frameJacobian(0);
      auto frame2_jacobian = workspace.frameJacobian(1);
      frame1_jacobian.setZero();
      frame2_jacobian.setZero();
      getFrameJacobian(
        model, data, cmodel.joint1_id, cmodel.joint1_placement, LOCAL, frame1_jacobian);
      getFrameJacobian(
        model, data, cmodel.joint2_id, cmodel.joint2_placement, LOCAL, frame2_jacobian);

      const SE3 & c1Mc2 = cdata.c1Mc2;
      const Motion frame1_velocity(frame1_jacobian * v);
      const Motion frame2_velocity(frame2_jacobian * v);
      const Motion frame2_velocity_in_frame1(c1Mc2.act(frame2_velocity));
      const Motion velocity_error(frame1_velocity - frame2_velocity_in_frame1);
      const Motion frame1_acceleration =
        getFrameAcceleration(model, data, cmodel.joint1_id, cmodel.joint1_placement, LOCAL);
      const Motion frame2_acceleration =
        getFrameAcceleration(model, data, cmodel.joint2_id, cmodel.joint2_placement, LOCAL);
      const Motion frame1_classical_acceleration = getFrameClassicalAcceleration(
        model, data, cmodel.joint1_id, cmodel.joint1_placement, LOCAL);
      const Motion frame2_classical_acceleration = getFrameClassicalAcceleration(
        model, data, cmodel.joint2_id, cmodel.joint2_placement, LOCAL);

      auto local_constraint_jacobian = workspace.localConstraintJacobian();
      for (Eigen::Index velocity_id = 0; velocity_id < model.nv; ++velocity_id)
      {
        const Motion frame1_jacobian_column(frame1_jacobian.col(velocity_id));
        const Motion frame2_jacobian_column(frame2_jacobian.col(velocity_id));
        if (cmodel.type == CONTACT_6D)
          local_constraint_jacobian.col(velocity_id) =
            (frame1_jacobian_column - c1Mc2.act(frame2_jacobian_column)).toVector();
        else
        {
          local_constraint_jacobian.template topRows<3>().col(velocity_id) =
            frame1_jacobian_column.linear() - c1Mc2.rotation() * frame2_jacobian_column.linear();
          local_constraint_jacobian.template bottomRows<3>().col(velocity_id).setZero();
        }
      }

      Matrix6 Jlog;
      Motion placement_error_local;
      if (cmodel.type == CONTACT_6D)
      {
        placement_error_local = -log6(c1Mc2);
        Jlog6(c1Mc2, Jlog);
      }
      else
      {
        placement_error_local.linear() = -c1Mc2.translation();
        placement_error_local.angular().setZero();
      }

      const SE3 output_rotation(cdata.oMc1.rotation(), Vector3::Zero());
      for (Eigen::Index parameter_id = 0; parameter_id < number_parameters; ++parameter_id)
      {
        const Motion frame1_tangent(frame1_placement_partial_dp.col(parameter_id));
        const Motion frame2_tangent(frame2_placement_partial_dp.col(parameter_id));
        Motion frame1_rotation_tangent(frame1_tangent);
        frame1_rotation_tangent.linear().setZero();
        const Motion relative_tangent(frame2_tangent - c1Mc2.actInv(frame1_tangent));
        relative_placement_partial_dp_.col(parameter_id) = relative_tangent.toVector();

        Motion placement_error_derivative_local;
        if (cmodel.type == CONTACT_6D)
          placement_error_derivative_local.toVector().noalias() =
            -Jlog * relative_tangent.toVector();
        else
        {
          placement_error_derivative_local.linear().noalias() =
            -c1Mc2.rotation() * relative_tangent.linear();
          placement_error_derivative_local.angular().setZero();
        }

        auto frame1_jacobian_derivative =
          frame1_jacobian_partial_dp.middleCols(parameter_id * (Eigen::Index)model.nv, model.nv);
        auto frame2_jacobian_derivative =
          frame2_jacobian_partial_dp.middleCols(parameter_id * (Eigen::Index)model.nv, model.nv);
        auto constraint_jacobian_derivative = constraint_jacobian_partial_dp_.middleCols(
          parameter_id * (Eigen::Index)model.nv, model.nv);
        for (Eigen::Index velocity_id = 0; velocity_id < model.nv; ++velocity_id)
        {
          const Motion frame1_jacobian_variation(frame1_jacobian_derivative.col(velocity_id));
          const Motion frame2_jacobian_variation(frame2_jacobian_derivative.col(velocity_id));
          Motion local_derivative;
          if (cmodel.type == CONTACT_6D)
            local_derivative =
              frame1_jacobian_variation
              - c1Mc2.act(
                frame2_jacobian_variation
                + relative_tangent.cross(Motion(frame2_jacobian.col(velocity_id))));
          else
          {
            local_derivative.linear() =
              frame1_jacobian_variation.linear()
              - c1Mc2.rotation()
                  * (frame2_jacobian_variation.linear()
                     + relative_tangent.angular().cross(
                       Motion(frame2_jacobian.col(velocity_id)).linear()));
            local_derivative.angular().setZero();
          }
          if (cmodel.reference_frame == LOCAL_WORLD_ALIGNED)
            local_derivative +=
              frame1_rotation_tangent.cross(Motion(local_constraint_jacobian.col(velocity_id)));

          const Motion output_derivative = cmodel.reference_frame == LOCAL
                                             ? local_derivative
                                             : output_rotation.act(local_derivative);
          if (cmodel.type == CONTACT_6D)
            constraint_jacobian_derivative.col(velocity_id) = output_derivative.toVector();
          else
            constraint_jacobian_derivative.col(velocity_id) = output_derivative.linear();
        }
        constraint_velocity_partial_dp_.col(parameter_id).noalias() =
          constraint_jacobian_derivative * v;

        Motion acceleration_derivative_local;
        if (cmodel.type == CONTACT_6D)
        {
          const Motion frame1_acceleration_derivative(
            frame1_acceleration_partial_dp.col(parameter_id));
          const Motion frame2_acceleration_derivative(
            frame2_acceleration_partial_dp.col(parameter_id));
          if (cmodel.reference_frame == LOCAL)
          {
            const Motion frame2_velocity_derivative(frame2_jacobian_derivative * v);
            const Motion frame2_velocity_variation_in_frame1(
              c1Mc2.act(frame2_velocity_derivative + relative_tangent.cross(frame2_velocity)));
            const Motion velocity_error_derivative(constraint_jacobian_derivative * v);
            acceleration_derivative_local =
              frame1_acceleration_derivative
              + velocity_error_derivative.cross(frame2_velocity_in_frame1)
              + velocity_error.cross(frame2_velocity_variation_in_frame1)
              - c1Mc2.act(
                frame2_acceleration_derivative + relative_tangent.cross(frame2_acceleration));
          }
          else
          {
            const Motion local_drift(frame1_acceleration - c1Mc2.act(frame2_acceleration));
            acceleration_derivative_local =
              frame1_acceleration_derivative
              - c1Mc2.act(
                frame2_acceleration_derivative + relative_tangent.cross(frame2_acceleration))
              + frame1_rotation_tangent.cross(local_drift);
          }
        }
        else
        {
          acceleration_derivative_local.linear() =
            Motion(frame1_classical_acceleration_partial_dp.col(parameter_id)).linear()
            - c1Mc2.rotation()
                * (Motion(frame2_classical_acceleration_partial_dp.col(parameter_id)).linear()
                   + relative_tangent.angular().cross(
                     frame2_classical_acceleration.linear()));
          acceleration_derivative_local.angular().setZero();
          if (cmodel.reference_frame == LOCAL_WORLD_ALIGNED)
          {
            const Vector3 local_drift = frame1_classical_acceleration.linear()
                                        - c1Mc2.rotation() * frame2_classical_acceleration.linear();
            acceleration_derivative_local.linear() +=
              frame1_rotation_tangent.angular().cross(local_drift);
          }
        }

        Motion placement_error_derivative = placement_error_derivative_local;
        Motion acceleration_derivative = acceleration_derivative_local;
        if (cmodel.reference_frame == LOCAL_WORLD_ALIGNED)
        {
          placement_error_derivative_local += frame1_rotation_tangent.cross(placement_error_local);
          placement_error_derivative = output_rotation.act(placement_error_derivative_local);
          acceleration_derivative = output_rotation.act(acceleration_derivative_local);
        }

        if (cmodel.type == CONTACT_6D)
        {
          placement_error_partial_dp_.col(parameter_id) = placement_error_derivative.toVector();
          constraint_acceleration_partial_dp_.col(parameter_id) =
            acceleration_derivative.toVector();
        }
        else
        {
          placement_error_partial_dp_.col(parameter_id) = placement_error_derivative.linear();
          constraint_acceleration_partial_dp_.col(parameter_id) = acceleration_derivative.linear();
        }
      }
    }
  } // namespace impl

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
    const Eigen::MatrixBase<AccelerationDerivativeType> & constraint_acceleration_partial_dp)
  {
    impl::computeRigidConstraintKinematicDerivatives(
      model, data, cmodel, cdata, make_const_ref(q), make_const_ref(v),
      make_const_ref(joint_placement_jacobians), workspace, make_ref(relative_placement_partial_dp),
      make_ref(placement_error_partial_dp), make_ref(constraint_jacobian_partial_dp),
      make_ref(constraint_velocity_partial_dp), make_ref(constraint_acceleration_partial_dp));
  }
} // namespace pinocchio

#ifdef PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION

namespace pinocchio
{
  namespace impl
  {
    extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void
    computeRigidConstraintKinematicDerivatives<
      context::Scalar,
      context::Options,
      JointCollectionDefaultTpl,
      Eigen::Ref<const context::VectorXs>,
      Eigen::Ref<const context::VectorXs>,
      Eigen::Ref<const context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>>(
      const Model &,
      Data &,
      const RigidConstraintModelTpl<context::Scalar, context::Options> &,
      RigidConstraintDataTpl<context::Scalar, context::Options> &,
      const Eigen::MatrixBase<Eigen::Ref<const context::VectorXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<const context::VectorXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<const context::MatrixXs>> &,
      RigidConstraintKinematicDerivativesWorkspaceTpl<context::Scalar, context::Options> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &);
  } // namespace impl
} // namespace pinocchio

#endif // PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION
