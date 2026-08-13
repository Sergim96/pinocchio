//
// Copyright (c) 2017-2019 CNRS INRIA
//

#pragma once

// IWYU pragma: begin_keep
#include <Eigen/Core>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <boost/fusion/container/vector.hpp>

#include "pinocchio/macros.hpp"
#include "pinocchio/eigen-common.hpp"
#include "pinocchio/fwd.hpp"

#include "pinocchio/math.hpp"

#include "pinocchio/spatial.hpp"

#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/joint.hpp"
#include "pinocchio/multibody/visitor.hpp"

#include "pinocchio/algorithm/check.hpp"
#include "pinocchio/algorithm/rnea.hpp"
// IWYU pragma: end_keep

namespace pinocchio
{

  ///
  /// \brief Reusable workspace for RNEA derivatives with respect to joint placements.
  ///
  /// The workspace stores four sets of spatial derivatives for every joint and parameter in one
  /// contiguous allocation. Construct it once and reuse it to keep
  /// computeRNEAPlacementDerivatives allocation-free.
  ///
  template<typename _Scalar, int _Options = 0>
  struct RNEAPlacementDerivativesWorkspaceTpl
  {
    typedef _Scalar Scalar;
    enum
    {
      Options = _Options
    };

    typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic, Options> Matrix6x;
    typedef Eigen::Index Index;

    RNEAPlacementDerivativesWorkspaceTpl()
    : storage(6, 0)
    , njoints(0)
    , parameter_capacity(0)
    {
    }

    RNEAPlacementDerivativesWorkspaceTpl(const Index njoints, const Index parameter_capacity)
    : storage(6, 0)
    , njoints(0)
    , parameter_capacity(0)
    {
      resize(njoints, parameter_capacity);
    }

    void resize(const Index new_njoints, const Index new_parameter_capacity)
    {
      assert(new_njoints >= 0);
      assert(new_parameter_capacity >= 0);
      njoints = new_njoints;
      parameter_capacity = new_parameter_capacity;
      storage.resize(6, 4 * njoints * parameter_capacity);
    }

    bool isCompatible(const Index expected_njoints, const Index number_parameters) const
    {
      return njoints == expected_njoints && parameter_capacity >= number_parameters;
    }

    auto relativePlacementDerivative(const Index joint_id, const Index number_parameters)
    {
      return block(0, joint_id, number_parameters);
    }

    auto velocityDerivative(const Index joint_id, const Index number_parameters)
    {
      return block(1, joint_id, number_parameters);
    }

    auto accelerationDerivative(const Index joint_id, const Index number_parameters)
    {
      return block(2, joint_id, number_parameters);
    }

    auto forceDerivative(const Index joint_id, const Index number_parameters)
    {
      return block(3, joint_id, number_parameters);
    }

    Matrix6x storage;
    Index njoints;
    Index parameter_capacity;

  protected:
    auto block(const Index set_id, const Index joint_id, const Index number_parameters)
    {
      assert(set_id >= 0 && set_id < 4);
      assert(joint_id >= 0 && joint_id < njoints);
      assert(number_parameters >= 0 && number_parameters <= parameter_capacity);
      const Index first_column = (set_id * njoints + joint_id) * parameter_capacity;
      return storage.middleCols(first_column, number_parameters);
    }
  };

  typedef RNEAPlacementDerivativesWorkspaceTpl<context::Scalar, context::Options>
    RNEAPlacementDerivativesWorkspace;

  ///
  /// \brief Computes RNEA derivatives with respect to right-trivialized joint-placement
  /// perturbations.
  ///
  /// The block joint_placement_jacobians.middleRows(6 * i, 6) is the right-trivialized
  /// derivative of model.jointPlacements[i]. The result has model.nv rows and one column per
  /// parameter. Derivatives of body inertias are not included.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in,out] data The data structure updated with the nominal RNEA quantities.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] a The joint acceleration vector (dim model.nv).
  /// \param[in] joint_placement_jacobians Stacked placement Jacobians with dimensions
  /// 6 * model.njoints by number_parameters. The universe block is ignored.
  /// \param[in,out] workspace Preallocated workspace compatible with the model and parameter count.
  /// \param[out] rnea_partial_dp Generalized torque derivative with dimensions model.nv by
  /// number_parameters.
  ///
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType1,
    typename TangentVectorType2,
    typename PlacementJacobianType,
    typename ReturnMatrixType>
  void computeRNEAPlacementDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType1> & v,
    const Eigen::MatrixBase<TangentVectorType2> & a,
    const Eigen::MatrixBase<PlacementJacobianType> & joint_placement_jacobians,
    RNEAPlacementDerivativesWorkspaceTpl<Scalar, Options> & workspace,
    const Eigen::MatrixBase<ReturnMatrixType> & rnea_partial_dp);

  ///
  /// \brief Computes the partial derivative of the generalized gravity contribution
  ///        with respect to the joint configuration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam ReturnMatrixType Type of the matrix containing the partial derivative of the gravity
  /// vector with respect to the joint configuration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[out] gravity_partial_dq Partial derivative of the generalized gravity vector with
  /// respect to the joint configuration.
  ///
  /// \remarks gravity_partial_dq must be first initialized with zeros (gravity_partial_dq.setZero).
  ///
  /// \sa pinocchio::computeGeneralizedGravity
  ///
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename ReturnMatrixType>
  void computeGeneralizedGravityDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<ReturnMatrixType> & gravity_partial_dq);

  ///
  /// \brief Computes the partial derivative of the generalized gravity and external forces
  /// contributions (a.k.a static torque vector)
  ///        with respect to the joint configuration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam ReturnMatrixType Type of the matrix containing the partial derivative of the gravity
  /// vector with respect to the joint configuration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] fext External forces expressed in the local frame of the joints (dim
  /// model.njoints). \param[out] static_torque_partial_dq Partial derivative of the static torque
  /// vector with respect to the joint configuration.
  ///
  /// \remarks gravity_partial_dq must be first initialized with zeros (gravity_partial_dq.setZero).
  ///
  /// \sa pinocchio::computeGeneralizedTorque
  ///
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename ReturnMatrixType>
  void computeStaticTorqueDerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const std::vector<ForceTpl<Scalar, Options>> & fext,
    const Eigen::MatrixBase<ReturnMatrixType> & static_torque_partial_dq);

  ///
  /// \brief Computes the partial derivatives of the Recursive Newton Euler Algorithms
  ///        with respect to the joint configuration, the joint velocity and the joint acceleration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  /// \tparam MatrixType1 Type of the matrix containing the partial derivative with respect to the
  /// joint configuration vector. \tparam MatrixType2 Type of the matrix containing the partial
  /// derivative with respect to the joint velocity vector. \tparam MatrixType3 Type of the matrix
  /// containing the partial derivative with respect to the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] a The joint acceleration vector (dim model.nv).
  /// \param[out] rnea_partial_dq Partial derivative of the generalized torque vector with respect
  /// to the joint configuration. \param[out] rnea_partial_dv Partial derivative of the generalized
  /// torque vector with respect to the joint velocity. \param[out] rnea_partial_da Partial
  /// derivative of the generalized torque vector with respect to the joint acceleration.
  ///
  /// \remarks rnea_partial_dq, rnea_partial_dv and rnea_partial_da must be first initialized with
  /// zeros (rnea_partial_dq.setZero(),etc).
  ///         As for pinocchio::crba, only the upper triangular part of rnea_partial_da is filled.
  ///
  /// \sa pinocchio::rnea
  ///
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType1,
    typename TangentVectorType2,
    typename MatrixType1,
    typename MatrixType2,
    typename MatrixType3>
  void computeRNEADerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType1> & v,
    const Eigen::MatrixBase<TangentVectorType2> & a,
    const Eigen::MatrixBase<MatrixType1> & rnea_partial_dq,
    const Eigen::MatrixBase<MatrixType2> & rnea_partial_dv,
    const Eigen::MatrixBase<MatrixType3> & rnea_partial_da);

  ///
  /// \brief Computes the derivatives of the Recursive Newton Euler Algorithms
  ///        with respect to the joint configuration, the joint velocity and the joint acceleration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  /// \tparam MatrixType1 Type of the matrix containing the partial derivative with respect to the
  /// joint configuration vector. \tparam MatrixType2 Type of the matrix containing the partial
  /// derivative with respect to the joint velocity vector. \tparam MatrixType3 Type of the matrix
  /// containing the partial derivative with respect to the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] a The joint acceleration vector (dim model.nv).
  /// \param[in] fext External forces expressed in the local frame of the joints (dim
  /// model.njoints). \param[out] rnea_partial_dq Partial derivative of the generalized torque
  /// vector with respect to the joint configuration. \param[out] rnea_partial_dv Partial derivative
  /// of the generalized torque vector with respect to the joint velocity. \param[out]
  /// rnea_partial_da Partial derivative of the generalized torque vector with respect to the joint
  /// acceleration.
  ///
  /// \remarks rnea_partial_dq, rnea_partial_dv and rnea_partial_da must be first initialized with
  /// zeros (rnea_partial_dq.setZero(),etc).
  ///         As for pinocchio::crba, only the upper triangular part of rnea_partial_da is filled.
  ///
  /// \sa pinocchio::rnea
  ///
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType1,
    typename TangentVectorType2,
    typename MatrixType1,
    typename MatrixType2,
    typename MatrixType3>
  void computeRNEADerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType1> & v,
    const Eigen::MatrixBase<TangentVectorType2> & a,
    const std::vector<ForceTpl<Scalar, Options>> & fext,
    const Eigen::MatrixBase<MatrixType1> & rnea_partial_dq,
    const Eigen::MatrixBase<MatrixType2> & rnea_partial_dv,
    const Eigen::MatrixBase<MatrixType3> & rnea_partial_da);

  ///
  /// \brief Computes the derivatives of the Recursive Newton Euler Algorithms
  ///        with respect to the joint configuration, the joint velocity and the joint acceleration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] a The joint acceleration vector (dim model.nv).
  ///
  /// \note The results are stored in data.dtau_dq, data.dtau_dv and data.M which respectively
  /// correspond
  ///          to the partial derivatives of the joint torque vector with respect to the joint
  ///          configuration, velocity and acceleration. As for pinocchio::crba, only the upper
  ///          triangular part of data.M is filled.
  ///
  /// \sa pinocchio::rnea, pinocchio::crba, pinocchio::decompose
  ///
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType1,
    typename TangentVectorType2>
  void computeRNEADerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType1> & v,
    const Eigen::MatrixBase<TangentVectorType2> & a);

  ///
  /// \brief Computes the derivatives of the Recursive Newton Euler Algorithms
  ///        with respect to the joint configuration, the joint velocity and the joint acceleration.
  ///
  /// \tparam JointCollection Collection of Joint types.
  /// \tparam ConfigVectorType Type of the joint configuration vector.
  /// \tparam TangentVectorType1 Type of the joint velocity vector.
  /// \tparam TangentVectorType2 Type of the joint acceleration vector.
  ///
  /// \param[in] model The model structure of the rigid body system.
  /// \param[in] data The data structure of the rigid body system.
  /// \param[in] q The joint configuration vector (dim model.nq).
  /// \param[in] v The joint velocity vector (dim model.nv).
  /// \param[in] a The joint acceleration vector (dim model.nv).
  /// \param[in] fext External forces expressed in the local frame of the joints (dim
  /// model.njoints).
  ///
  /// \note The results are stored in data.dtau_dq, data.dtau_dv and data.M which respectively
  /// correspond
  ///          to the partial derivatives of the joint torque vector with respect to the joint
  ///          configuration, velocity and acceleration. As for pinocchio::crba, only the upper
  ///          triangular part of data.M is filled.
  ///
  /// \sa pinocchio::rnea, pinocchio::crba, pinocchio::decompose
  ///
  template<
    typename Scalar,
    int Options,
    template<typename, int> class JointCollectionTpl,
    typename ConfigVectorType,
    typename TangentVectorType1,
    typename TangentVectorType2>
  void computeRNEADerivatives(
    const ModelTpl<Scalar, Options, JointCollectionTpl> & model,
    DataTpl<Scalar, Options, JointCollectionTpl> & data,
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<TangentVectorType1> & v,
    const Eigen::MatrixBase<TangentVectorType2> & a,
    const std::vector<ForceTpl<Scalar, Options>> & fext);

} // namespace pinocchio

// IWYU pragma: begin_exports
#include "pinocchio/src/algorithm/rnea-derivatives.hxx"
// IWYU pragma: end_exports
