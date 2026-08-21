//
// Copyright (c) 2020 INRIA
//

#pragma once

// IWYU pragma: begin_keep
#include <Eigen/Core>

#include <cassert>
#include <cstddef>
#include <vector>

#include <boost/fusion/container/vector.hpp>

#include "pinocchio/macros.hpp"
#include "pinocchio/eigen-common.hpp"

#include "pinocchio/math.hpp"

#include "pinocchio/spatial.hpp"

#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/joint.hpp"
#include "pinocchio/multibody/visitor.hpp"

#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
// IWYU pragma: end_keep

namespace pinocchio
{

  ///
  /// \brief Reusable workspace for frame derivatives with respect to joint placements.
  ///
  /// Construct it once and reuse it to keep computeFramePlacementDerivatives allocation-free.
  /// The parameter capacity may be larger than the number of parameter directions used by a call.
  ///
  template<typename _Scalar, int _Options = 0>
  struct FramePlacementDerivativesWorkspaceTpl
  {
    typedef _Scalar Scalar;
    enum
    {
      Options = _Options
    };

    typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic, Options> Matrix6x;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options> VectorXs;
    typedef Eigen::Index Index;

    FramePlacementDerivativesWorkspaceTpl()
    : parameter_storage(6, 0)
    , jacobian_storage(6, 0)
    , zero_acceleration(0)
    , nv(0)
    , parameter_capacity(0)
    , current_set(0)
    {
    }

    FramePlacementDerivativesWorkspaceTpl(const Index nv, const Index parameter_capacity)
    : parameter_storage(6, 0)
    , jacobian_storage(6, 0)
    , zero_acceleration(0)
    , nv(0)
    , parameter_capacity(0)
    , current_set(0)
    {
      resize(nv, parameter_capacity);
    }

    void resize(const Index new_nv, const Index new_parameter_capacity)
    {
      assert(new_nv >= 0);
      assert(new_parameter_capacity >= 0);
      nv = new_nv;
      parameter_capacity = new_parameter_capacity;
      parameter_storage.resize(6, 7 * parameter_capacity);
      jacobian_storage.resize(6, 2 * nv);
      zero_acceleration.setZero(nv);
      current_set = 0;
    }

    bool isCompatible(const Index expected_nv, const Index number_parameters) const
    {
      return nv == expected_nv && parameter_capacity >= number_parameters;
    }

    auto relativePlacementDerivative(const Index number_parameters)
    {
      return parameterBlock(0, number_parameters);
    }

    auto placementDerivative(const Index set_id, const Index number_parameters)
    {
      return parameterBlock(1 + set_id, number_parameters);
    }

    auto velocityDerivative(const Index set_id, const Index number_parameters)
    {
      return parameterBlock(3 + set_id, number_parameters);
    }

    auto accelerationDerivative(const Index set_id, const Index number_parameters)
    {
      return parameterBlock(5 + set_id, number_parameters);
    }

    auto jacobian(const Index set_id)
    {
      assert(set_id >= 0 && set_id < 2);
      return jacobian_storage.middleCols(set_id * nv, nv);
    }

    Matrix6x parameter_storage;
    Matrix6x jacobian_storage;
    VectorXs zero_acceleration;
    Index nv;
    Index parameter_capacity;
    Index current_set;

  protected:
    auto parameterBlock(const Index block_id, const Index number_parameters)
    {
      assert(block_id >= 0 && block_id < 7);
      assert(number_parameters >= 0 && number_parameters <= parameter_capacity);
      return parameter_storage.middleCols(block_id * parameter_capacity, number_parameters);
    }
  };

  typedef FramePlacementDerivativesWorkspaceTpl<context::Scalar, context::Options>
    FramePlacementDerivativesWorkspace;

  ///
  /// \brief Computes frame Jacobian and drift derivatives with respect to right-trivialized
  /// joint-placement perturbations.
  ///
  /// This operation differentiates one frame relative to the universe. The block
  /// joint_placement_jacobians may contain either all model.njoints blocks, with an ignored
  /// universe block, or only the model.njoints - 1 non-universe blocks. The block for joint i is
  /// the right-trivialized derivative of model.jointPlacements[i]. The Jacobian derivative is
  /// flattened parameter-major: middleCols(p * model.nv, model.nv) contains the 6 by model.nv
  /// derivative for parameter p.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in,out] data Data updated with the nominal frame kinematics.
  /// \param[in] q Joint configuration vector (dim model.nq).
  /// \param[in] v Joint velocity vector (dim model.nv).
  /// \param[in] frame_id Index of the differentiated frame.
  /// \param[in] joint_placement_jacobians Stacked placement Jacobians with either
  /// 6 * model.njoints or 6 * (model.njoints - 1) rows and number_parameters columns.
  /// \param[in] reference_frame Reference frame in which the results are expressed.
  /// \param[in,out] workspace Reusable workspace compatible with model.nv and the parameter count.
  /// \param[out] frame_placement_partial_dp Right-trivialized frame placement derivative in LOCAL
  /// coordinates, with dimensions 6 by number_parameters.
  /// \param[out] frame_jacobian_partial_dp Flattened frame Jacobian derivative with dimensions
  /// 6 by model.nv * number_parameters.
  /// \param[out] frame_acceleration_partial_dp Spatial drift derivative with dimensions 6 by
  /// number_parameters.
  /// \param[out] frame_classical_acceleration_partial_dp Classical drift derivative with dimensions
  /// 6 by number_parameters.
  ///
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
      frame_classical_acceleration_partial_dp);

  /**
   * @brief      Computes the partial derivatives of the spatial velocity of a frame given by its
   * relative placement, with respect to q and v. You must first call
   * pinocchio::computeForwardKinematicsDerivatives to compute all the required quantities.
   *
   * @tparam JointCollection Collection of Joint types.
   * @tparam Matrix6xOut1 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint configuration vector.
   * @tparam Matrix6xOut2 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint velocity vector.
   *
   * @param[in]  model                   The kinematic model
   * @param[in]  data                     Data associated to model
   * @param[in]  joint_id            Index of the supporting joint
   * @param[in]  placement          Placement of the Frame w.r.t. the joint frame.
   * @param[in]  rf                         Reference frame in which the velocity is expressed.
   * @param[out] v_partial_dq   Partial derivative of the frame spatial velocity w.r.t. \f$ q \f$.
   * @param[out] v_partial_dv   Partial derivative of the frame spatial velociy w.r.t. \f$ v \f$.
   *
   */
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
    const Eigen::MatrixBase<Matrix6xOut2> & v_partial_dv);

  /**
   * @brief      Computes the partial derivatives of the frame spatial velocity with respect to q
   * and v. You must first call pinocchio::computeForwardKinematicsDerivatives to compute all the
   * required quantities.
   *
   * @tparam JointCollection Collection of Joint types.
   * @tparam Matrix6xOut1 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint configuration vector.
   * @tparam Matrix6xOut2 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint velocity vector.
   *
   * @param[in]  model       The kinematic model
   * @param[in]  data        Data associated to model
   * @param[in]  frame_id    Id of the operational Frame
   * @param[in]  rf          Reference frame in which the velocity is expressed.
   * @param[out] v_partial_dq Partial derivative of the frame spatial velocity w.r.t. \f$ q \f$.
   * @param[out] v_partial_dv Partial derivative of the frame spatial velociy w.r.t. \f$ v \f$.
   *
   */
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename Matrix6xOut1,
    typename Matrix6xOut2>
  void getFrameVelocityDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const FrameIndex frame_id,
    const ReferenceFrame rf,
    const Eigen::MatrixBase<Matrix6xOut1> & v_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut2> & v_partial_dv)
  {
    PINOCCHIO_CHECK_INPUT_ARGUMENT((int)frame_id < model.nframes, "The frame_id is not valid.");
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;
    typedef typename Model::Frame Frame;

    const Frame & frame = model.frames[frame_id];
    typename Data::SE3 & oMframe = data.oMf[frame_id];
    oMframe = data.oMi[frame.parentJoint] * frame.placement; // for backward compatibility
    getFrameVelocityDerivatives(
      model, data, frame.parentJoint, frame.placement, rf,
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut1, v_partial_dq),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut1, v_partial_dv));
  }

  /**
   * @brief      Computes the partial derivatives of the spatial acceleration of a frame given by
   * its relative placement, with respect to q, v and a. You must first call
   * pinocchio::computeForwardKinematicsDerivatives to compute all the required quantities. It is
   * important to notice that a direct outcome (for free) of this algo is v_partial_dq and
   * v_partial_dv which is equal to a_partial_da.
   *
   * @tparam JointCollection Collection of Joint types.
   * @tparam Matrix6xOut1 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint configuration vector.
   * @tparam Matrix6xOut2 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint configuration vector.
   * @tparam Matrix6xOut3 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint velocity vector.
   * @tparam Matrix6xOut4 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint acceleration vector.
   *
   * @param[in]  model                   The kinematic model
   * @param[in]  data                     Data associated to model
   * @param[in]  joint_id            Index of the supporting joint
   * @param[in]  placement          Placement of the Frame w.r.t. the joint frame.
   * @param[in]  rf                          Reference frame in which the velocity is expressed.
   * @param[out] v_partial_dq    Partial derivative of the frame spatial velocity w.r.t. \f$ q \f$.
   * @param[out] a_partial_dq    Partial derivative of the frame spatial acceleration w.r.t. \f$ q
   * \f$.
   * @param[out] a_partial_dv    Partial derivative of the frame spatial acceleration w.r.t. \f$ v
   * \f$.
   * @param[out] a_partial_da    Partial derivative of the frame spatial acceleration w.r.t. \f$
   * \dot{v} \f$.
   *
   */
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
    const Eigen::MatrixBase<Matrix6xOut4> & a_partial_da);

  /**
   * @brief      Computes the partial derivatives of the frame acceleration quantity with respect to
   * q, v and a. You must first call pinocchio::computeForwardKinematicsDerivatives to compute all
   * the required quantities. It is important to notice that a direct outcome (for free) of this
   * algo is v_partial_dq and v_partial_dv which is equal to a_partial_da.
   *
   * @tparam JointCollection Collection of Joint types.
   * @tparam Matrix6xOut1 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint configuration vector.
   * @tparam Matrix6xOut2 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint configuration vector.
   * @tparam Matrix6xOut3 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint velocity vector.
   * @tparam Matrix6xOut4 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint acceleration vector.
   *
   * @param[in]  model       The kinematic model
   * @param[in]  data        Data associated to model
   * @param[in]  frame_id    Id of the operational Frame
   * @param[in]  rf          Reference frame in which the velocity is expressed.
   * @param[out] v_partial_dq Partial derivative of the frame spatial velocity w.r.t. \f$ q \f$.
   * @param[out] a_partial_dq Partial derivative of the frame spatial acceleration w.r.t. \f$ q \f$.
   * @param[out] a_partial_dv Partial derivative of the frame spatial acceleration w.r.t. \f$ v \f$.
   * @param[out] a_partial_da Partial derivative of the frame spatial acceleration w.r.t. \f$
   * \dot{v} \f$.
   *
   */
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
    const FrameIndex frame_id,
    const ReferenceFrame rf,
    const Eigen::MatrixBase<Matrix6xOut1> & v_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut2> & a_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut3> & a_partial_dv,
    const Eigen::MatrixBase<Matrix6xOut4> & a_partial_da)
  {
    PINOCCHIO_CHECK_INPUT_ARGUMENT((int)frame_id < model.nframes, "The frame_id is not valid.");
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;
    typedef typename Model::Frame Frame;

    const Frame & frame = model.frames[frame_id];
    typename Data::SE3 & oMframe = data.oMf[frame_id];
    oMframe = data.oMi[frame.parentJoint] * frame.placement; // for backward compatibility
    getFrameAccelerationDerivatives(
      model, data, frame.parentJoint, frame.placement, rf,
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut1, v_partial_dq),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut2, a_partial_dq),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut3, a_partial_dv),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut4, a_partial_da));
  }

  /**
   * @brief      Computes the partial derivatives of the frame acceleration quantity with respect to
   * q, v and a. You must first call pinocchio::computeForwardKinematicsDerivatives to compute all
   * the required quantities. It is important to notice that a direct outcome (for free) of this
   * algo is v_partial_dq and v_partial_dv which is equal to a_partial_da.
   *
   * @tparam JointCollection Collection of Joint types.
   * @tparam Matrix6xOut1 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint configuration vector.
   * @tparam Matrix6xOut2 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint velocity vector.
   * @tparam Matrix6xOut3 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint configuration vector.
   * @tparam Matrix6xOut4 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint velocity vector.
   * @tparam Matrix6xOut5 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint acceleration vector.
   *
   * @param[in]  model                  The kinematic model
   * @param[in]  data                     Data associated to model
   * @param[in]  joint_id            Index of the supporting joint
   * @param[in]  placement          Placement of the Frame w.r.t. the joint frame.
   * @param[in]  rf                          Reference frame in which the velocity is expressed.
   * @param[out] v_partial_dq   Partial derivative of the frame spatial velocity w.r.t. \f$ q \f$.
   * @param[out] v_partial_dv   Partial derivative of the frame spatial velociy w.r.t. \f$ v \f$.
   * @param[out] a_partial_dq   Partial derivative of the frame spatial acceleration w.r.t. \f$ q
   * \f$.
   * @param[out] a_partial_dv   Partial derivative of the frame spatial acceleration w.r.t. \f$ v
   * \f$.
   * @param[out] a_partial_da   Partial derivative of the frame spatial acceleration w.r.t. \f$
   * \dot{v} \f$.
   *
   */
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename Matrix6xOut1,
    typename Matrix6xOut2,
    typename Matrix6xOut3,
    typename Matrix6xOut4,
    typename Matrix6xOut5>
  void getFrameAccelerationDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const JointIndex joint_id,
    const SE3Tpl<Scalar, Options> & placement,
    const ReferenceFrame rf,
    const Eigen::MatrixBase<Matrix6xOut1> & v_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut2> & v_partial_dv,
    const Eigen::MatrixBase<Matrix6xOut3> & a_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut4> & a_partial_dv,
    const Eigen::MatrixBase<Matrix6xOut5> & a_partial_da)
  {
    getFrameAccelerationDerivatives(
      model, data, joint_id, placement, rf, PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut1, v_partial_dq),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut3, a_partial_dq),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut4, a_partial_dv),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut5, a_partial_da));

    PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut2, v_partial_dv) = a_partial_da;
  }

  /**
   * @brief      Computes the partial derivatives of the frame acceleration quantity with respect to
   * q, v and a. You must first call pinocchio::computeForwardKinematicsDerivatives to compute all
   * the required quantities. It is important to notice that a direct outcome (for free) of this
   * algo is v_partial_dq and v_partial_dv which is equal to a_partial_da.
   *
   * @tparam JointCollection Collection of Joint types.
   * @tparam Matrix6xOut1 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint configuration vector.
   * @tparam Matrix6xOut2 Matrix6x containing the partial derivatives of the frame spatial velocity
   * with respect to the joint velocity vector.
   * @tparam Matrix6xOut3 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint configuration vector.
   * @tparam Matrix6xOut4 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint velocity vector.
   * @tparam Matrix6xOut5 Matrix6x containing the partial derivatives of the frame spatial
   * acceleration with respect to the joint acceleration vector.
   *
   * @param[in]  model       The kinematic model
   * @param[in]  data        Data associated to model
   * @param[in]  frame_id    Id of the operational Frame
   * @param[in]  rf          Reference frame in which the velocity is expressed.
   * @param[out] v_partial_dq Partial derivative of the frame spatial velocity w.r.t. \f$ q \f$.
   * @param[out] v_partial_dv Partial derivative of the frame spatial velociy w.r.t. \f$ v \f$.
   * @param[out] a_partial_dq Partial derivative of the frame spatial acceleration w.r.t. \f$ q \f$.
   * @param[out] a_partial_dv Partial derivative of the frame spatial acceleration w.r.t. \f$ v \f$.
   * @param[out] a_partial_da Partial derivative of the frame spatial acceleration w.r.t. \f$
   * \dot{v} \f$.
   *
   */
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename Matrix6xOut1,
    typename Matrix6xOut2,
    typename Matrix6xOut3,
    typename Matrix6xOut4,
    typename Matrix6xOut5>
  void getFrameAccelerationDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const FrameIndex frame_id,
    const ReferenceFrame rf,
    const Eigen::MatrixBase<Matrix6xOut1> & v_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut2> & v_partial_dv,
    const Eigen::MatrixBase<Matrix6xOut3> & a_partial_dq,
    const Eigen::MatrixBase<Matrix6xOut4> & a_partial_dv,
    const Eigen::MatrixBase<Matrix6xOut5> & a_partial_da)
  {
    PINOCCHIO_CHECK_INPUT_ARGUMENT((int)frame_id < model.nframes, "The frame_id is not valid.");
    typedef ModelTpl<Scalar, Options, JointCollectionTpl> Model;
    typedef DataTpl<Scalar, Options, JointCollectionTpl> Data;
    typedef typename Model::Frame Frame;

    const Frame & frame = model.frames[frame_id];
    typename Data::SE3 & oMframe = data.oMf[frame_id];
    oMframe = data.oMi[frame.parentJoint] * frame.placement; // for backward compatibility
    getFrameAccelerationDerivatives(
      model, data, frame.parentJoint, frame.placement, rf,
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut1, v_partial_dq),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut2, v_partial_dv),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut3, a_partial_dq),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut4, a_partial_dv),
      PINOCCHIO_EIGEN_CONST_CAST(Matrix6xOut5, a_partial_da));
  }
} // namespace pinocchio

// IWYU pragma: begin_exports
#include "pinocchio/src/algorithm/frames-derivatives.hxx"
// IWYU pragma: end_exports
