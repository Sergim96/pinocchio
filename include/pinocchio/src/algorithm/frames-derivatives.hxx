//
// Copyright (c) 2020 INRIA
//

#pragma once

// IWYU pragma: private, include "pinocchio/algorithm/frames-derivatives.hpp"

#ifdef PINOCCHIO_LSP
  #undef PINOCCHIO_LSP
  #include "pinocchio/algorithm/frames-derivatives.hpp"
#endif // PINOCCHIO_LSP

namespace pinocchio
{
  namespace impl
  {
    template<
      typename Scalar,
      int Options,
      template<typename, int> class JointCollectionTpl,
      typename PlacementJacobianType,
      typename JacobianDerivativeType>
    struct ComputeFramePlacementDerivativesForwardStep
    : public fusion::JointUnaryVisitorBase<ComputeFramePlacementDerivativesForwardStep<
        Scalar,
        Options,
        JointCollectionTpl,
        PlacementJacobianType,
        JacobianDerivativeType>>
    {
      typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
      typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;
      typedef FramePlacementDerivativesWorkspaceTpl<Scalar, Options> Workspace;

      typedef boost::fusion::vector<
        const Model &,
        Data &,
        const PlacementJacobianType &,
        Workspace &,
        JacobianDerivativeType &>
        ArgsType;

      template<typename JointModel>
      static void algo(
        const JointModelBase<JointModel> & jmodel,
        JointDataBase<typename JointModel::JointDataDerived> & jdata,
        const Model & model,
        Data & data,
        const Eigen::MatrixBase<PlacementJacobianType> & joint_placement_jacobians,
        Workspace & workspace,
        const Eigen::MatrixBase<JacobianDerivativeType> & frame_jacobian_partial_dp)
      {
        typedef typename Model::JointIndex JointIndex;
        typedef typename Data::Motion Motion;
        typedef typename Data::SE3 SE3;

        const JointIndex i = jmodel.id();
        const JointIndex parent = model.parents[i];
        const Eigen::Index number_parameters = joint_placement_jacobians.cols();
        const Eigen::Index current_set = workspace.current_set;
        const Eigen::Index next_set = 1 - current_set;

        auto relative_placement_derivatives =
          workspace.relativePlacementDerivative(number_parameters);
        auto placement_derivatives = workspace.placementDerivative(current_set, number_parameters);
        auto next_placement_derivatives =
          workspace.placementDerivative(next_set, number_parameters);
        auto velocity_derivatives = workspace.velocityDerivative(current_set, number_parameters);
        auto next_velocity_derivatives = workspace.velocityDerivative(next_set, number_parameters);
        auto acceleration_derivatives =
          workspace.accelerationDerivative(current_set, number_parameters);
        auto next_acceleration_derivatives =
          workspace.accelerationDerivative(next_set, number_parameters);

        const Eigen::Index joint_row =
          joint_placement_jacobians.rows() == 6 * (Eigen::Index)model.njoints
            ? 6 * (Eigen::Index)i
            : 6 * ((Eigen::Index)i - 1);
        const auto placement_jacobian =
          joint_placement_jacobians.derived().middleRows(joint_row, 6);
        const SE3 joint_motion_placement(jdata.M());
        motionSet::se3ActionInverse(
          joint_motion_placement, placement_jacobian, relative_placement_derivatives);

        motionSet::se3ActionInverse(
          data.liMi[i], placement_derivatives, next_placement_derivatives);
        next_placement_derivatives += relative_placement_derivatives;

        motionSet::se3ActionInverse(data.liMi[i], velocity_derivatives, next_velocity_derivatives);
        const Motion parent_velocity(data.liMi[i].actInv(data.v[parent]));
        motionSet::motionAction<ADDTO>(
          parent_velocity, relative_placement_derivatives, next_velocity_derivatives);

        motionSet::se3ActionInverse(
          data.liMi[i], acceleration_derivatives, next_acceleration_derivatives);
        const Motion parent_acceleration(data.liMi[i].actInv(data.a[parent]));
        motionSet::motionAction<ADDTO>(
          parent_acceleration, relative_placement_derivatives, next_acceleration_derivatives);
        const Motion joint_velocity(jdata.v());
        motionSet::motionAction<RMTO>(
          joint_velocity, next_velocity_derivatives, next_acceleration_derivatives);

        auto parent_jacobian = workspace.jacobian(current_set);
        auto current_jacobian = workspace.jacobian(next_set);
        motionSet::se3ActionInverse(data.liMi[i], parent_jacobian, current_jacobian);

        JacobianDerivativeType & frame_jacobian_partial_dp_ =
          PINOCCHIO_EIGEN_CONST_CAST(JacobianDerivativeType, frame_jacobian_partial_dp);
        for (Eigen::Index parameter_id = 0; parameter_id < number_parameters; ++parameter_id)
        {
          auto jacobian_derivative =
            frame_jacobian_partial_dp_.middleCols(parameter_id * (Eigen::Index)model.nv, model.nv);
          for (Eigen::Index velocity_id = 0; velocity_id < model.nv; ++velocity_id)
          {
            const Motion derivative(jacobian_derivative.col(velocity_id));
            jacobian_derivative.col(velocity_id) = data.liMi[i].actInv(derivative).toVector();
          }
          const Motion relative_placement_derivative(
            relative_placement_derivatives.col(parameter_id));
          motionSet::motionAction<RMTO>(
            relative_placement_derivative, current_jacobian, jacobian_derivative);
        }
        jmodel.jointCols(current_jacobian) += jdata.S().matrix();

        workspace.current_set = next_set;
      }
    };

    template<
      typename Scalar,
      int Options,
      template<typename, int> class JointCollectionTpl,
      typename PlacementJacobianType,
      typename PlacementDerivativeType,
      typename JacobianDerivativeType,
      typename AccelerationDerivativeType,
      typename ClassicalAccelerationDerivativeType>
    void computeFramePlacementDerivativesPrepared(
      const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
      DataTpl<Scalar, Options, JointCollectionTpl> & data,
      const JointIndex joint_id,
      const SE3Tpl<Scalar, Options> & placement,
      const SE3Tpl<Scalar, Options> & oMframe,
      const Eigen::MatrixBase<PlacementJacobianType> & joint_placement_jacobians,
      const ReferenceFrame reference_frame,
      FramePlacementDerivativesWorkspaceTpl<Scalar, Options> & workspace,
      const Eigen::MatrixBase<PlacementDerivativeType> & frame_placement_partial_dp,
      const Eigen::MatrixBase<JacobianDerivativeType> & frame_jacobian_partial_dp,
      const Eigen::MatrixBase<AccelerationDerivativeType> & frame_acceleration_partial_dp,
      const Eigen::MatrixBase<ClassicalAccelerationDerivativeType> &
        frame_classical_acceleration_partial_dp)
    {
      typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
      typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;
      typedef typename Model::JointIndex JointIndex;
      typedef typename Data::Motion Motion;
      typedef typename Data::SE3 SE3;
      typedef typename Data::Vector3 Vector3;

      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        joint_id < (JointIndex)model.njoints, "The joint_id is not valid.");
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        reference_frame == LOCAL || reference_frame == LOCAL_WORLD_ALIGNED
          || reference_frame == WORLD,
        "The reference frame is not valid.");
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        joint_placement_jacobians.rows() == 6 * model.njoints
          || joint_placement_jacobians.rows() == 6 * (model.njoints - 1),
        "The placement Jacobian must have 6 * model.njoints or 6 * (model.njoints - 1) rows");

      const Eigen::Index number_parameters = joint_placement_jacobians.cols();
      PINOCCHIO_CHECK_ARGUMENT_SIZE(frame_placement_partial_dp.rows(), 6);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(frame_placement_partial_dp.cols(), number_parameters);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(frame_jacobian_partial_dp.rows(), 6);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(frame_jacobian_partial_dp.cols(), model.nv * number_parameters);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(frame_acceleration_partial_dp.rows(), 6);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(frame_acceleration_partial_dp.cols(), number_parameters);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(frame_classical_acceleration_partial_dp.rows(), 6);
      PINOCCHIO_CHECK_ARGUMENT_SIZE(
        frame_classical_acceleration_partial_dp.cols(), number_parameters);
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        workspace.isCompatible(model.nv, number_parameters),
        "The frame placement-derivative workspace is not compatible with model.nv or the parameter "
        "count");
      assert(model.check(data) && "data is not consistent with model.");

      JacobianDerivativeType & frame_jacobian_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(JacobianDerivativeType, frame_jacobian_partial_dp);
      PlacementDerivativeType & frame_placement_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(PlacementDerivativeType, frame_placement_partial_dp);
      AccelerationDerivativeType & frame_acceleration_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(AccelerationDerivativeType, frame_acceleration_partial_dp);
      ClassicalAccelerationDerivativeType & frame_classical_acceleration_partial_dp_ =
        PINOCCHIO_EIGEN_CONST_CAST(
          ClassicalAccelerationDerivativeType, frame_classical_acceleration_partial_dp);

      frame_jacobian_partial_dp_.setZero();
      workspace.current_set = 0;
      workspace.placementDerivative(0, number_parameters).setZero();
      workspace.velocityDerivative(0, number_parameters).setZero();
      workspace.accelerationDerivative(0, number_parameters).setZero();
      workspace.jacobian(0).setZero();

      typedef ComputeFramePlacementDerivativesForwardStep<
        Scalar, Options, JointCollectionTpl, PlacementJacobianType, JacobianDerivativeType>
        Pass;
      typename Pass::ArgsType args(
        model, data, joint_placement_jacobians.derived(), workspace, frame_jacobian_partial_dp_);
      const typename Model::IndexVector & support = model.supports[joint_id];
      for (std::size_t support_id = 1; support_id < support.size(); ++support_id)
      {
        const JointIndex joint_id = support[support_id];
        Pass::run(model.joints[joint_id], data.joints[joint_id], args);
      }

      const Eigen::Index current_set = workspace.current_set;
      const Eigen::Index next_set = 1 - current_set;
      auto placement_derivatives = workspace.placementDerivative(current_set, number_parameters);
      auto frame_placement_derivatives = workspace.placementDerivative(next_set, number_parameters);
      auto velocity_derivatives = workspace.velocityDerivative(current_set, number_parameters);
      auto frame_velocity_derivatives = workspace.velocityDerivative(next_set, number_parameters);
      auto acceleration_derivatives =
        workspace.accelerationDerivative(current_set, number_parameters);
      auto frame_acceleration_derivatives =
        workspace.accelerationDerivative(next_set, number_parameters);

      motionSet::se3ActionInverse(placement, placement_derivatives, frame_placement_derivatives);
      frame_placement_partial_dp_ = frame_placement_derivatives;
      motionSet::se3ActionInverse(placement, velocity_derivatives, frame_velocity_derivatives);
      motionSet::se3ActionInverse(
        placement, acceleration_derivatives, frame_acceleration_derivatives);

      auto parent_jacobian = workspace.jacobian(current_set);
      auto frame_jacobian = workspace.jacobian(next_set);
      motionSet::se3ActionInverse(placement, parent_jacobian, frame_jacobian);
      for (Eigen::Index parameter_id = 0; parameter_id < number_parameters; ++parameter_id)
      {
        auto jacobian_derivative =
          frame_jacobian_partial_dp_.middleCols(parameter_id * (Eigen::Index)model.nv, model.nv);
        for (Eigen::Index velocity_id = 0; velocity_id < model.nv; ++velocity_id)
        {
          const Motion derivative(jacobian_derivative.col(velocity_id));
          jacobian_derivative.col(velocity_id) = placement.actInv(derivative).toVector();
        }
      }

      Motion frame_velocity = pinocchio::getFrameVelocity(model, data, joint_id, placement, LOCAL);
      const Motion frame_acceleration =
        pinocchio::getFrameAcceleration(model, data, joint_id, placement, LOCAL);

      if (reference_frame != LOCAL)
      {
        const SE3 output_placement =
          reference_frame == WORLD ? oMframe : SE3(oMframe.rotation(), Vector3::Zero());

        for (Eigen::Index parameter_id = 0; parameter_id < number_parameters; ++parameter_id)
        {
          Motion frame_tangent(frame_placement_derivatives.col(parameter_id));
          if (reference_frame == LOCAL_WORLD_ALIGNED)
            frame_tangent.linear().setZero();

          auto jacobian_derivative =
            frame_jacobian_partial_dp_.middleCols(parameter_id * (Eigen::Index)model.nv, model.nv);
          motionSet::motionAction<ADDTO>(frame_tangent, frame_jacobian, jacobian_derivative);
          for (Eigen::Index velocity_id = 0; velocity_id < model.nv; ++velocity_id)
          {
            const Motion derivative(jacobian_derivative.col(velocity_id));
            jacobian_derivative.col(velocity_id) = output_placement.act(derivative).toVector();
          }

          Motion velocity_derivative(frame_velocity_derivatives.col(parameter_id));
          velocity_derivative += frame_tangent.cross(frame_velocity);
          frame_velocity_derivatives.col(parameter_id) =
            output_placement.act(velocity_derivative).toVector();

          Motion acceleration_derivative(frame_acceleration_derivatives.col(parameter_id));
          acceleration_derivative += frame_tangent.cross(frame_acceleration);
          frame_acceleration_derivatives.col(parameter_id) =
            output_placement.act(acceleration_derivative).toVector();
        }
        frame_velocity = output_placement.act(frame_velocity);
      }

      frame_acceleration_partial_dp_ = frame_acceleration_derivatives;
      frame_classical_acceleration_partial_dp_ = frame_acceleration_derivatives;
      for (Eigen::Index parameter_id = 0; parameter_id < number_parameters; ++parameter_id)
      {
        const Motion velocity_derivative(frame_velocity_derivatives.col(parameter_id));
        frame_classical_acceleration_partial_dp_.template topRows<3>().col(parameter_id) +=
          velocity_derivative.angular().cross(frame_velocity.linear())
          + frame_velocity.angular().cross(velocity_derivative.linear());
      }
    }

    template<
      typename Scalar,
      int Options,
      template<typename, int> class JointCollectionTpl,
      typename ConfigVectorType,
      typename TangentVectorType,
      typename PlacementJacobianType,
      typename PlacementDerivativeType,
      typename JacobianDerivativeType,
      typename AccelerationDerivativeType,
      typename ClassicalAccelerationDerivativeType>
    void computeFramePlacementDerivatives(
      const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
      DataTpl<Scalar, Options, JointCollectionTpl> & data,
      const Eigen::MatrixBase<ConfigVectorType> & q,
      const Eigen::MatrixBase<TangentVectorType> & v,
      const FrameIndex frame_id,
      const Eigen::MatrixBase<PlacementJacobianType> & joint_placement_jacobians,
      const ReferenceFrame reference_frame,
      FramePlacementDerivativesWorkspaceTpl<Scalar, Options> & workspace,
      const Eigen::MatrixBase<PlacementDerivativeType> & frame_placement_partial_dp,
      const Eigen::MatrixBase<JacobianDerivativeType> & frame_jacobian_partial_dp,
      const Eigen::MatrixBase<AccelerationDerivativeType> & frame_acceleration_partial_dp,
      const Eigen::MatrixBase<ClassicalAccelerationDerivativeType> &
        frame_classical_acceleration_partial_dp)
    {
      PINOCCHIO_CHECK_ARGUMENT_SIZE(
        q.size(), model.nq, "The joint configuration vector is not of right size");
      PINOCCHIO_CHECK_ARGUMENT_SIZE(
        v.size(), model.nv, "The joint velocity vector is not of right size");
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
        frame_id < (FrameIndex)model.nframes, "The frame_id is not valid.");

      pinocchio::forwardKinematics(model, data, q, v, workspace.zero_acceleration);
      pinocchio::computeJointJacobians(model, data);
      pinocchio::updateFramePlacement(model, data, frame_id);

      const typename ModelTpl<Scalar, Options, JointCollectionTpl>::Frame & frame =
        model.frames[frame_id];
      computeFramePlacementDerivativesPrepared(
        model, data, frame.parentJoint, frame.placement, data.oMf[frame_id],
        joint_placement_jacobians, reference_frame, workspace, frame_placement_partial_dp,
        frame_jacobian_partial_dp, frame_acceleration_partial_dp,
        frame_classical_acceleration_partial_dp);
    }
  } // namespace impl

  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType,
    typename PlacementJacobianType,
    typename PlacementDerivativeType,
    typename JacobianDerivativeType,
    typename AccelerationDerivativeType,
    typename ClassicalAccelerationDerivativeType>
  void computeFramePlacementDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType> & v,
    const FrameIndex frame_id,
    const Eigen::MatrixBase<PlacementJacobianType> & joint_placement_jacobians,
    const ReferenceFrame reference_frame,
    FramePlacementDerivativesWorkspaceTpl<Scalar, Options> & workspace,
    const Eigen::MatrixBase<PlacementDerivativeType> & frame_placement_partial_dp,
    const Eigen::MatrixBase<JacobianDerivativeType> & frame_jacobian_partial_dp,
    const Eigen::MatrixBase<AccelerationDerivativeType> & frame_acceleration_partial_dp,
    const Eigen::MatrixBase<ClassicalAccelerationDerivativeType> &
      frame_classical_acceleration_partial_dp)
  {
    impl::computeFramePlacementDerivatives(
      model, data, make_const_ref(q), make_const_ref(v), frame_id,
      make_const_ref(joint_placement_jacobians), reference_frame, workspace,
      make_ref(frame_placement_partial_dp), make_ref(frame_jacobian_partial_dp),
      make_ref(frame_acceleration_partial_dp), make_ref(frame_classical_acceleration_partial_dp));
  }

  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename Matrix6xOut1,
    typename Matrix6xOut2>
  void getFrameVelocityDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    const DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const JointIndex joint_id,
    const SE3Tpl<Scalar, Options> & placement,
    const ReferenceFrame rf,
    const Eigen::MatrixBase<Matrix6xOut1> & v_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut2> & v_partial_dv)
  {
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

    typedef typename Data::Matrix6x Matrix6x;
    typedef typename Data::SE3 SE3;
    typedef typename Data::Motion Motion;

    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Matrix6xOut1, Matrix6x);
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Matrix6xOut2, Matrix6x);

    Matrix6xOut1 & v_partial_dq_ = PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut1, v_partial_dq);
    Matrix6xOut2 & v_partial_dv_ = PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut2, v_partial_dv);
    getJointVelocityDerivatives(model, data, joint_id, rf, v_partial_dq_, v_partial_dv_);

    typedef typename SizeDepType<1>::template ColsReturn<Matrix6xOut1>::Type ColsBlockOut1;
    typedef MotionRef<ColsBlockOut1> MotionOut1;
    typedef typename SizeDepType<1>::template ColsReturn<Matrix6xOut2>::Type ColsBlockOut2;
    typedef MotionRef<ColsBlockOut2> MotionOut2;

    Motion v_tmp;
    const typename SE3::Vector3 trans = data.oMi[joint_id].rotation() * placement.translation();
    const int colRef = nv(model.joints[joint_id]) + idx_v(model.joints[joint_id]) - 1;
    switch (rf)
    {
    case WORLD:
      // Do nothing
      break;

    case LOCAL_WORLD_ALIGNED:
      for (Eigen::Index col_id = colRef; col_id >= 0; col_id = data.parents_fromRow[(size_t)col_id])
      {
        MotionOut1 m1(v_partial_dq_.col(col_id));
        m1.linear() -= trans.cross(m1.angular());
        MotionOut2 m2(v_partial_dv_.col(col_id));
        m2.linear() -= trans.cross(m2.angular());
      }
      break;

    case LOCAL:
      for (Eigen::Index col_id = colRef; col_id >= 0; col_id = data.parents_fromRow[(size_t)col_id])
      {
        v_tmp = v_partial_dq_.col(col_id);
        MotionOut1(v_partial_dq_.col(col_id)) = placement.actInv(v_tmp);
        v_tmp = v_partial_dv_.col(col_id);
        MotionOut2(v_partial_dv_.col(col_id)) = placement.actInv(v_tmp);
      }
      break;

    default:
      break;
    }
  }

  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename Matrix6xOut1,
    typename Matrix6xOut2,
    typename Matrix6xOut3,
    typename Matrix6xOut4>
  void getFrameAccelerationDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const JointIndex joint_id,
    const SE3Tpl<Scalar, Options> & placement,
    const ReferenceFrame rf,
    const Eigen::MatrixBase<Matrix6xOut1> & v_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut2> & a_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut3> & a_partial_dv,
    const Eigen::MatrixBase<Matrix6xOut4> & a_partial_da)
  {
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;

    typedef typename Data::Matrix6x Matrix6x;
    typedef typename Data::SE3 SE3;
    typedef typename Data::Motion Motion;

    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Matrix6xOut1, Matrix6x);
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Matrix6xOut2, Matrix6x);
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Matrix6xOut3, Matrix6x);
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Matrix6xOut4, Matrix6x);

    PINOCCHIO_CHECK_ARGUMENT_SIZE(v_partial_dq.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a_partial_dq.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a_partial_dv.cols(), model.nv);
    PINOCCHIO_CHECK_ARGUMENT_SIZE(a_partial_da.cols(), model.nv);
    assert(model.check(data) && "data is not consistent with model.");

    Matrix6xOut1 & v_partial_dq_ = PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut1, v_partial_dq);
    Matrix6xOut2 & a_partial_dq_ = PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut2, a_partial_dq);
    Matrix6xOut3 & a_partial_dv_ = PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut3, a_partial_dv);
    Matrix6xOut4 & a_partial_da_ = PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut4, a_partial_da);

    getJointAccelerationDerivatives(
      model, data, joint_id, rf, v_partial_dq_, a_partial_dq_, a_partial_dv_, a_partial_da_);

    typedef typename SizeDepType<1>::template ColsReturn<Matrix6xOut1>::Type ColsBlockOut1;
    typedef MotionRef<ColsBlockOut1> MotionOut1;
    typedef typename SizeDepType<1>::template ColsReturn<Matrix6xOut2>::Type ColsBlockOut2;
    typedef MotionRef<ColsBlockOut2> MotionOut2;
    typedef typename SizeDepType<1>::template ColsReturn<Matrix6xOut3>::Type ColsBlockOut3;
    typedef MotionRef<ColsBlockOut3> MotionOut3;
    typedef typename SizeDepType<1>::template ColsReturn<Matrix6xOut4>::Type ColsBlockOut4;
    typedef MotionRef<ColsBlockOut4> MotionOut4;

    const int colRef = nv(model.joints[joint_id]) + idx_v(model.joints[joint_id]) - 1;
    switch (rf)
    {
    case WORLD:
      // Do nothing
      break;

    case LOCAL_WORLD_ALIGNED: {
      const typename SE3::Vector3 trans = data.oMi[joint_id].rotation() * placement.translation();
      for (Eigen::Index col_id = colRef; col_id >= 0; col_id = data.parents_fromRow[(size_t)col_id])
      {
        MotionOut1 m1(v_partial_dq_.col(col_id));
        m1.linear() -= trans.cross(m1.angular());
        MotionOut2 m2(a_partial_dq_.col(col_id));
        m2.linear() -= trans.cross(m2.angular());
        MotionOut3 m3(a_partial_dv_.col(col_id));
        m3.linear() -= trans.cross(m3.angular());
        MotionOut4 m4(a_partial_da_.col(col_id));
        m4.linear() -= trans.cross(m4.angular());
      }
      break;
    }
    case LOCAL: {
      Motion v_tmp;
      for (Eigen::Index col_id = colRef; col_id >= 0; col_id = data.parents_fromRow[(size_t)col_id])
      {
        v_tmp = v_partial_dq_.col(col_id);
        MotionOut1(v_partial_dq_.col(col_id)) = placement.actInv(v_tmp);
        v_tmp = a_partial_dq_.col(col_id);
        MotionOut2(a_partial_dq_.col(col_id)) = placement.actInv(v_tmp);
        v_tmp = a_partial_dv_.col(col_id);
        MotionOut3(a_partial_dv_.col(col_id)) = placement.actInv(v_tmp);
        v_tmp = a_partial_da_.col(col_id);
        MotionOut4(a_partial_da_.col(col_id)) = placement.actInv(v_tmp);
      }
      break;
    }
    default:
      break;
    }
  }
} // namespace pinocchio

#ifdef PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION

namespace pinocchio
{
  namespace impl
  {
    extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void
    computeFramePlacementDerivatives<
      context::Scalar,
      context::Options,
      JointCollectionDefaultTpl,
      Eigen::Ref<const context::VectorXs>,
      Eigen::Ref<const context::VectorXs>,
      Eigen::Ref<const context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>,
      Eigen::Ref<context::MatrixXs>>(
      const Model &,
      Data &,
      const Eigen::MatrixBase<Eigen::Ref<const context::VectorXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<const context::VectorXs>> &,
      const FrameIndex,
      const Eigen::MatrixBase<Eigen::Ref<const context::MatrixXs>> &,
      const ReferenceFrame,
      FramePlacementDerivativesWorkspaceTpl<context::Scalar, context::Options> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &,
      const Eigen::MatrixBase<Eigen::Ref<context::MatrixXs>> &);
  } // namespace impl

  extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void
  getFrameVelocityDerivatives<
    context::Scalar,
    context::Options,
    JointCollectionDefaultTpl,
    context::Matrix6xs,
    context::Matrix6xs>(
    const Model &,
    const Data &,
    const JointIndex,
    const SE3Tpl<context::Scalar, context::Options> &,
    const ReferenceFrame,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &);

  extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void
  getFrameVelocityDerivatives<
    context::Scalar,
    context::Options,
    JointCollectionDefaultTpl,
    context::Matrix6xs,
    context::Matrix6xs>(
    const Model &,
    Data &,
    const FrameIndex,
    const ReferenceFrame,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &);

  extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void
  getFrameAccelerationDerivatives<
    context::Scalar,
    context::Options,
    JointCollectionDefaultTpl,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs>(
    const Model &,
    Data &,
    const JointIndex,
    const SE3Tpl<context::Scalar, context::Options> &,
    const ReferenceFrame,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &);

  extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void
  getFrameAccelerationDerivatives<
    context::Scalar,
    context::Options,
    JointCollectionDefaultTpl,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs>(
    const Model &,
    Data &,
    const FrameIndex,
    const ReferenceFrame,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &);

  extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void
  getFrameAccelerationDerivatives<
    context::Scalar,
    context::Options,
    JointCollectionDefaultTpl,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs>(
    const Model &,
    Data &,
    const JointIndex,
    const SE3Tpl<context::Scalar, context::Options> &,
    const ReferenceFrame,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &);

  extern template PINOCCHIO_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI void
  getFrameAccelerationDerivatives<
    context::Scalar,
    context::Options,
    JointCollectionDefaultTpl,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs,
    context::Matrix6xs>(
    const Model &,
    Data &,
    const FrameIndex,
    const ReferenceFrame,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &,
    const Eigen::MatrixBase<context::Matrix6xs> &);
} // namespace pinocchio
#endif // ifdef PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION
