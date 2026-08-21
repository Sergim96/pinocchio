//
// Copyright (c) 2022 INRIA
//

#include "pinocchio/src/context/template-instantiation.hxx"
#include "pinocchio/algorithm/frames-derivatives.hpp"

namespace pinocchio
{
  namespace impl
  {
    template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void
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

  template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void getFrameVelocityDerivatives<
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

  template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void getFrameVelocityDerivatives<
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

  template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void getFrameAccelerationDerivatives<
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

  template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void getFrameAccelerationDerivatives<
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

  template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void getFrameAccelerationDerivatives<
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

  template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void getFrameAccelerationDerivatives<
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
