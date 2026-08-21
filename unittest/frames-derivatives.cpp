//
// Copyright (c) 2020 INRIA
//

#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_frames_derivatives_velocity)
{
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model);

  const Model::JointIndex jointId = model.existJointName("rarm2_joint")
                                      ? model.getJointId("rarm2_joint")
                                      : (Model::Index)(model.njoints - 1);
  Frame frame("rand", jointId, 0, SE3::Random(), OP_FRAME);
  FrameIndex frameId = model.addFrame(frame);

  BOOST_CHECK(model.getFrameId("rand") == frameId);
  BOOST_CHECK(model.frames[frameId].parentJoint == jointId);

  Data data(model), data_ref(model);

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill(1.);
  VectorXd q = randomConfiguration(model);
  VectorXd v(VectorXd::Random(model.nv));
  VectorXd a(VectorXd::Random(model.nv));

  computeForwardKinematicsDerivatives(model, data, q, v, a);

  Data::Matrix6x partial_dq(6, model.nv);
  partial_dq.setZero();
  Data::Matrix6x partial_dq_local_world_aligned(6, model.nv);
  partial_dq_local_world_aligned.setZero();
  Data::Matrix6x partial_dq_local(6, model.nv);
  partial_dq_local.setZero();
  Data::Matrix6x partial_dv(6, model.nv);
  partial_dv.setZero();
  Data::Matrix6x partial_dv_local_world_aligned(6, model.nv);
  partial_dv_local_world_aligned.setZero();
  Data::Matrix6x partial_dv_local(6, model.nv);
  partial_dv_local.setZero();

  getFrameVelocityDerivatives(model, data, frameId, WORLD, partial_dq, partial_dv);

  getFrameVelocityDerivatives(
    model, data, frameId, LOCAL_WORLD_ALIGNED, partial_dq_local_world_aligned,
    partial_dv_local_world_aligned);

  getFrameVelocityDerivatives(model, data, frameId, LOCAL, partial_dq_local, partial_dv_local);

  Data::Matrix6x J_ref(6, model.nv);
  J_ref.setZero();
  Data::Matrix6x J_ref_local_world_aligned(6, model.nv);
  J_ref_local_world_aligned.setZero();
  Data::Matrix6x J_ref_local(6, model.nv);
  J_ref_local.setZero();
  computeJointJacobians(model, data_ref, q);
  getFrameJacobian(model, data_ref, frameId, WORLD, J_ref);
  getFrameJacobian(model, data_ref, frameId, LOCAL_WORLD_ALIGNED, J_ref_local_world_aligned);
  getFrameJacobian(model, data_ref, frameId, LOCAL, J_ref_local);

  BOOST_CHECK(data_ref.oMf[frameId].isApprox(data.oMf[frameId]));
  BOOST_CHECK(partial_dv.isApprox(J_ref));
  BOOST_CHECK(partial_dv_local_world_aligned.isApprox(J_ref_local_world_aligned));
  BOOST_CHECK(partial_dv_local.isApprox(J_ref_local));

  // Check against finite differences
  Data::Matrix6x partial_dq_fd(6, model.nv);
  partial_dq_fd.setZero();
  Data::Matrix6x partial_dq_fd_local_world_aligned(6, model.nv);
  partial_dq_fd_local_world_aligned.setZero();
  Data::Matrix6x partial_dq_fd_local(6, model.nv);
  partial_dq_fd_local.setZero();
  Data::Matrix6x partial_dv_fd(6, model.nv);
  partial_dv_fd.setZero();
  Data::Matrix6x partial_dv_fd_local_world_aligned(6, model.nv);
  partial_dv_fd_local_world_aligned.setZero();
  Data::Matrix6x partial_dv_fd_local(6, model.nv);
  partial_dv_fd_local.setZero();
  const double alpha = 1e-8;

  // dvel/dv
  Eigen::VectorXd v_plus(v);
  Data data_plus(model);
  forwardKinematics(model, data_ref, q, v);
  Motion v0 = getFrameVelocity(model, data, frameId, WORLD);
  Motion v0_local_world_aligned = getFrameVelocity(model, data, frameId, LOCAL_WORLD_ALIGNED);
  Motion v0_local = getFrameVelocity(model, data, frameId, LOCAL);
  for (int k = 0; k < model.nv; ++k)
  {
    v_plus[k] += alpha;
    forwardKinematics(model, data_plus, q, v_plus);

    partial_dv_fd.col(k) =
      (getFrameVelocity(model, data_plus, frameId, WORLD) - v0).toVector() / alpha;
    partial_dv_fd_local_world_aligned.col(k) =
      (getFrameVelocity(model, data_plus, frameId, LOCAL_WORLD_ALIGNED) - v0_local_world_aligned)
        .toVector()
      / alpha;
    partial_dv_fd_local.col(k) =
      (getFrameVelocity(model, data_plus, frameId, LOCAL) - v0_local).toVector() / alpha;
    v_plus[k] -= alpha;
  }

  BOOST_CHECK(partial_dv.isApprox(partial_dv_fd, sqrt(alpha)));
  BOOST_CHECK(
    partial_dv_local_world_aligned.isApprox(partial_dv_fd_local_world_aligned, sqrt(alpha)));
  BOOST_CHECK(partial_dv_local.isApprox(partial_dv_fd_local, sqrt(alpha)));

  // dvel/dq
  Eigen::VectorXd q_plus(q), v_eps(Eigen::VectorXd::Zero(model.nv));
  forwardKinematics(model, data_ref, q, v);
  updateFramePlacements(model, data_ref);

  for (int k = 0; k < model.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(model, q, v_eps);
    forwardKinematics(model, data_plus, q_plus, v);
    updateFramePlacements(model, data_plus);

    Motion v_plus_local_world_aligned =
      getFrameVelocity(model, data_plus, frameId, LOCAL_WORLD_ALIGNED);
    SE3::Vector3 trans = data_plus.oMf[frameId].translation() - data_ref.oMf[frameId].translation();
    v_plus_local_world_aligned.linear() -= v_plus_local_world_aligned.angular().cross(trans);
    partial_dq_fd.col(k) =
      (getFrameVelocity(model, data_plus, frameId, WORLD) - v0).toVector() / alpha;
    partial_dq_fd_local_world_aligned.col(k) =
      (v_plus_local_world_aligned - v0_local_world_aligned).toVector() / alpha;
    partial_dq_fd_local.col(k) =
      (getFrameVelocity(model, data_plus, frameId, LOCAL) - v0_local).toVector() / alpha;
    v_eps[k] -= alpha;
  }

  BOOST_CHECK(partial_dq.isApprox(partial_dq_fd, sqrt(alpha)));
  BOOST_CHECK(
    partial_dq_local_world_aligned.isApprox(partial_dq_fd_local_world_aligned, sqrt(alpha)));
  BOOST_CHECK(partial_dq_local.isApprox(partial_dq_fd_local, sqrt(alpha)));
}

BOOST_AUTO_TEST_CASE(test_kinematics_derivatives_acceleration)
{
  using namespace Eigen;
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model);

  const Model::JointIndex jointId = model.existJointName("rarm2_joint")
                                      ? model.getJointId("rarm2_joint")
                                      : (Model::Index)(model.njoints - 1);
  Frame frame("rand", jointId, 0, SE3::Random(), OP_FRAME);
  FrameIndex frameId = model.addFrame(frame);

  BOOST_CHECK(model.getFrameId("rand") == frameId);
  BOOST_CHECK(model.frames[frameId].parentJoint == jointId);

  Data data(model), data_ref(model);

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill(1.);
  VectorXd q = randomConfiguration(model);
  VectorXd v(VectorXd::Random(model.nv));
  VectorXd a(VectorXd::Random(model.nv));

  computeForwardKinematicsDerivatives(model, data, q, v, a);

  Data::Matrix6x v_partial_dq(6, model.nv);
  v_partial_dq.setZero();
  Data::Matrix6x v_partial_dq_local(6, model.nv);
  v_partial_dq_local.setZero();
  Data::Matrix6x v_partial_dq_local_world_aligned(6, model.nv);
  v_partial_dq_local_world_aligned.setZero();
  Data::Matrix6x a_partial_dq(6, model.nv);
  a_partial_dq.setZero();
  Data::Matrix6x a_partial_dq_local_world_aligned(6, model.nv);
  a_partial_dq_local_world_aligned.setZero();
  Data::Matrix6x a_partial_dq_local(6, model.nv);
  a_partial_dq_local.setZero();
  Data::Matrix6x a_partial_dv(6, model.nv);
  a_partial_dv.setZero();
  Data::Matrix6x a_partial_dv_local_world_aligned(6, model.nv);
  a_partial_dv_local_world_aligned.setZero();
  Data::Matrix6x a_partial_dv_local(6, model.nv);
  a_partial_dv_local.setZero();
  Data::Matrix6x a_partial_da(6, model.nv);
  a_partial_da.setZero();
  Data::Matrix6x a_partial_da_local_world_aligned(6, model.nv);
  a_partial_da_local_world_aligned.setZero();
  Data::Matrix6x a_partial_da_local(6, model.nv);
  a_partial_da_local.setZero();

  getFrameAccelerationDerivatives(
    model, data, frameId, WORLD, v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da);

  getFrameAccelerationDerivatives(
    model, data, frameId, LOCAL_WORLD_ALIGNED, v_partial_dq_local_world_aligned,
    a_partial_dq_local_world_aligned, a_partial_dv_local_world_aligned,
    a_partial_da_local_world_aligned);

  getFrameAccelerationDerivatives(
    model, data, frameId, LOCAL, v_partial_dq_local, a_partial_dq_local, a_partial_dv_local,
    a_partial_da_local);

  // Check v_partial_dq against getFrameVelocityDerivatives
  {
    Data data_v(model);
    computeForwardKinematicsDerivatives(model, data_v, q, v, a);

    Data::Matrix6x v_partial_dq_ref(6, model.nv);
    v_partial_dq_ref.setZero();
    Data::Matrix6x v_partial_dq_ref_local_world_aligned(6, model.nv);
    v_partial_dq_ref_local_world_aligned.setZero();
    Data::Matrix6x v_partial_dq_ref_local(6, model.nv);
    v_partial_dq_ref_local.setZero();
    Data::Matrix6x v_partial_dv_ref(6, model.nv);
    v_partial_dv_ref.setZero();
    Data::Matrix6x v_partial_dv_ref_local_world_aligned(6, model.nv);
    v_partial_dv_ref_local_world_aligned.setZero();
    Data::Matrix6x v_partial_dv_ref_local(6, model.nv);
    v_partial_dv_ref_local.setZero();

    getFrameVelocityDerivatives(model, data_v, frameId, WORLD, v_partial_dq_ref, v_partial_dv_ref);

    BOOST_CHECK(v_partial_dq.isApprox(v_partial_dq_ref));
    BOOST_CHECK(a_partial_da.isApprox(v_partial_dv_ref));

    getFrameVelocityDerivatives(
      model, data_v, frameId, LOCAL_WORLD_ALIGNED, v_partial_dq_ref_local_world_aligned,
      v_partial_dv_ref_local_world_aligned);

    BOOST_CHECK(v_partial_dq_local_world_aligned.isApprox(v_partial_dq_ref_local_world_aligned));
    BOOST_CHECK(a_partial_da_local_world_aligned.isApprox(v_partial_dv_ref_local_world_aligned));

    getFrameVelocityDerivatives(
      model, data_v, frameId, LOCAL, v_partial_dq_ref_local, v_partial_dv_ref_local);

    BOOST_CHECK(v_partial_dq_local.isApprox(v_partial_dq_ref_local));
    BOOST_CHECK(a_partial_da_local.isApprox(v_partial_dv_ref_local));
  }

  Data::Matrix6x J_ref(6, model.nv);
  J_ref.setZero();
  Data::Matrix6x J_ref_local(6, model.nv);
  J_ref_local.setZero();
  Data::Matrix6x J_ref_local_world_aligned(6, model.nv);
  J_ref_local_world_aligned.setZero();
  computeJointJacobians(model, data_ref, q);
  getFrameJacobian(model, data_ref, frameId, WORLD, J_ref);
  getFrameJacobian(model, data_ref, frameId, LOCAL_WORLD_ALIGNED, J_ref_local_world_aligned);
  getFrameJacobian(model, data_ref, frameId, LOCAL, J_ref_local);

  BOOST_CHECK(a_partial_da.isApprox(J_ref));
  BOOST_CHECK(a_partial_da_local_world_aligned.isApprox(J_ref_local_world_aligned));
  BOOST_CHECK(a_partial_da_local.isApprox(J_ref_local));

  // Check against finite differences
  Data::Matrix6x a_partial_da_fd(6, model.nv);
  a_partial_da_fd.setZero();
  Data::Matrix6x a_partial_da_fd_local_world_aligned(6, model.nv);
  a_partial_da_fd_local_world_aligned.setZero();
  Data::Matrix6x a_partial_da_fd_local(6, model.nv);
  a_partial_da_fd_local.setZero();
  const double alpha = 1e-8;

  Eigen::VectorXd v_plus(v), a_plus(a);
  Data data_plus(model);
  forwardKinematics(model, data_ref, q, v, a);

  // dacc/da
  Motion a0 = getFrameAcceleration(model, data, frameId, WORLD);
  Motion a0_local_world_aligned = getFrameAcceleration(model, data, frameId, LOCAL_WORLD_ALIGNED);
  Motion a0_local = getFrameAcceleration(model, data, frameId, LOCAL);
  for (int k = 0; k < model.nv; ++k)
  {
    a_plus[k] += alpha;
    forwardKinematics(model, data_plus, q, v, a_plus);

    a_partial_da_fd.col(k) =
      (getFrameAcceleration(model, data_plus, frameId, WORLD) - a0).toVector() / alpha;
    a_partial_da_fd_local_world_aligned.col(k) =
      (getFrameAcceleration(model, data_plus, frameId, LOCAL_WORLD_ALIGNED)
       - a0_local_world_aligned)
        .toVector()
      / alpha;
    a_partial_da_fd_local.col(k) =
      (getFrameAcceleration(model, data_plus, frameId, LOCAL) - a0_local).toVector() / alpha;
    a_plus[k] -= alpha;
  }
  BOOST_CHECK(a_partial_da.isApprox(a_partial_da_fd, sqrt(alpha)));
  BOOST_CHECK(
    a_partial_da_local_world_aligned.isApprox(a_partial_da_fd_local_world_aligned, sqrt(alpha)));
  BOOST_CHECK(a_partial_da_local.isApprox(a_partial_da_fd_local, sqrt(alpha)));

  // dacc/dv
  Data::Matrix6x a_partial_dv_fd(6, model.nv);
  a_partial_dv_fd.setZero();
  Data::Matrix6x a_partial_dv_fd_local_world_aligned(6, model.nv);
  a_partial_dv_fd_local_world_aligned.setZero();
  Data::Matrix6x a_partial_dv_fd_local(6, model.nv);
  a_partial_dv_fd_local.setZero();
  for (int k = 0; k < model.nv; ++k)
  {
    v_plus[k] += alpha;
    forwardKinematics(model, data_plus, q, v_plus, a);

    a_partial_dv_fd.col(k) =
      (getFrameAcceleration(model, data_plus, frameId, WORLD) - a0).toVector() / alpha;
    a_partial_dv_fd_local_world_aligned.col(k) =
      (getFrameAcceleration(model, data_plus, frameId, LOCAL_WORLD_ALIGNED)
       - a0_local_world_aligned)
        .toVector()
      / alpha;
    a_partial_dv_fd_local.col(k) =
      (getFrameAcceleration(model, data_plus, frameId, LOCAL) - a0_local).toVector() / alpha;
    v_plus[k] -= alpha;
  }

  BOOST_CHECK(a_partial_dv.isApprox(a_partial_dv_fd, sqrt(alpha)));
  BOOST_CHECK(
    a_partial_dv_local_world_aligned.isApprox(a_partial_dv_fd_local_world_aligned, sqrt(alpha)));
  BOOST_CHECK(a_partial_dv_local.isApprox(a_partial_dv_fd_local, sqrt(alpha)));

  // dacc/dq
  a_partial_dq.setZero();
  a_partial_dv.setZero();
  a_partial_da.setZero();

  a_partial_dq_local_world_aligned.setZero();
  a_partial_dv_local_world_aligned.setZero();
  a_partial_da_local_world_aligned.setZero();

  a_partial_dq_local.setZero();
  a_partial_dv_local.setZero();
  a_partial_da_local.setZero();

  Data::Matrix6x a_partial_dq_fd(6, model.nv);
  a_partial_dq_fd.setZero();
  Data::Matrix6x a_partial_dq_fd_local_world_aligned(6, model.nv);
  a_partial_dq_fd_local_world_aligned.setZero();
  Data::Matrix6x a_partial_dq_fd_local(6, model.nv);
  a_partial_dq_fd_local.setZero();

  computeForwardKinematicsDerivatives(model, data, q, v, a);
  getFrameAccelerationDerivatives(
    model, data, frameId, WORLD, v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da);

  getFrameAccelerationDerivatives(
    model, data, frameId, LOCAL_WORLD_ALIGNED, v_partial_dq_local_world_aligned,
    a_partial_dq_local_world_aligned, a_partial_dv_local_world_aligned,
    a_partial_da_local_world_aligned);

  getFrameAccelerationDerivatives(
    model, data, frameId, LOCAL, v_partial_dq_local, a_partial_dq_local, a_partial_dv_local,
    a_partial_da_local);

  Eigen::VectorXd q_plus(q), v_eps(Eigen::VectorXd::Zero(model.nv));
  forwardKinematics(model, data_ref, q, v, a);
  updateFramePlacements(model, data_ref);
  a0 = getFrameAcceleration(model, data, frameId, WORLD);
  a0_local_world_aligned = getFrameAcceleration(model, data, frameId, LOCAL_WORLD_ALIGNED);
  a0_local = getFrameAcceleration(model, data, frameId, LOCAL);

  for (int k = 0; k < model.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(model, q, v_eps);
    forwardKinematics(model, data_plus, q_plus, v, a);
    updateFramePlacements(model, data_plus);

    a_partial_dq_fd.col(k) =
      (getFrameAcceleration(model, data_plus, frameId, WORLD) - a0).toVector() / alpha;
    Motion a_plus_local_world_aligned =
      getFrameAcceleration(model, data_plus, frameId, LOCAL_WORLD_ALIGNED);
    const SE3::Vector3 trans =
      data_plus.oMf[frameId].translation() - data_ref.oMf[frameId].translation();
    a_plus_local_world_aligned.linear() -= a_plus_local_world_aligned.angular().cross(trans);
    a_partial_dq_fd_local_world_aligned.col(k) =
      (a_plus_local_world_aligned - a0_local_world_aligned).toVector() / alpha;
    a_partial_dq_fd_local.col(k) =
      (getFrameAcceleration(model, data_plus, frameId, LOCAL) - a0_local).toVector() / alpha;
    v_eps[k] -= alpha;
  }

  BOOST_CHECK(a_partial_dq.isApprox(a_partial_dq_fd, sqrt(alpha)));
  BOOST_CHECK(
    a_partial_dq_local_world_aligned.isApprox(a_partial_dq_fd_local_world_aligned, sqrt(alpha)));
  BOOST_CHECK(a_partial_dq_local.isApprox(a_partial_dq_fd_local, sqrt(alpha)));

  // Test other signatures
  Data::Matrix6x v_partial_dq_other(6, model.nv);
  v_partial_dq_other.setZero();
  Data::Matrix6x v_partial_dq_local_other(6, model.nv);
  v_partial_dq_local_other.setZero();
  Data::Matrix6x v_partial_dq_local_world_aligned_other(6, model.nv);
  v_partial_dq_local_world_aligned_other.setZero();
  Data::Matrix6x v_partial_dv_other(6, model.nv);
  v_partial_dv_other.setZero();
  Data::Matrix6x v_partial_dv_local_other(6, model.nv);
  v_partial_dv_local_other.setZero();
  Data::Matrix6x v_partial_dv_local_world_aligned_other(6, model.nv);
  v_partial_dv_local_world_aligned_other.setZero();
  Data::Matrix6x a_partial_dq_other(6, model.nv);
  a_partial_dq_other.setZero();
  Data::Matrix6x a_partial_dq_local_world_aligned_other(6, model.nv);
  a_partial_dq_local_world_aligned_other.setZero();
  Data::Matrix6x a_partial_dq_local_other(6, model.nv);
  a_partial_dq_local_other.setZero();
  Data::Matrix6x a_partial_dv_other(6, model.nv);
  a_partial_dv_other.setZero();
  Data::Matrix6x a_partial_dv_local_world_aligned_other(6, model.nv);
  a_partial_dv_local_world_aligned_other.setZero();
  Data::Matrix6x a_partial_dv_local_other(6, model.nv);
  a_partial_dv_local_other.setZero();
  Data::Matrix6x a_partial_da_other(6, model.nv);
  a_partial_da_other.setZero();
  Data::Matrix6x a_partial_da_local_world_aligned_other(6, model.nv);
  a_partial_da_local_world_aligned_other.setZero();
  Data::Matrix6x a_partial_da_local_other(6, model.nv);
  a_partial_da_local_other.setZero();

  getFrameAccelerationDerivatives(
    model, data, frameId, WORLD, v_partial_dq_other, v_partial_dv_other, a_partial_dq_other,
    a_partial_dv_other, a_partial_da_other);

  BOOST_CHECK(v_partial_dq_other.isApprox(v_partial_dq));
  BOOST_CHECK(v_partial_dv_other.isApprox(a_partial_da));
  BOOST_CHECK(a_partial_dq_other.isApprox(a_partial_dq));
  BOOST_CHECK(a_partial_dv_other.isApprox(a_partial_dv));
  BOOST_CHECK(a_partial_da_other.isApprox(a_partial_da));

  getFrameAccelerationDerivatives(
    model, data, frameId, LOCAL_WORLD_ALIGNED, v_partial_dq_local_world_aligned_other,
    v_partial_dv_local_world_aligned_other, a_partial_dq_local_world_aligned_other,
    a_partial_dv_local_world_aligned_other, a_partial_da_local_world_aligned_other);

  BOOST_CHECK(v_partial_dq_local_world_aligned_other.isApprox(v_partial_dq_local_world_aligned));
  BOOST_CHECK(v_partial_dv_local_world_aligned_other.isApprox(a_partial_da_local_world_aligned));
  BOOST_CHECK(a_partial_dq_local_world_aligned_other.isApprox(a_partial_dq_local_world_aligned));
  BOOST_CHECK(a_partial_dv_local_world_aligned_other.isApprox(a_partial_dv_local_world_aligned));
  BOOST_CHECK(a_partial_da_local_world_aligned_other.isApprox(a_partial_da_local_world_aligned));

  getFrameAccelerationDerivatives(
    model, data, frameId, LOCAL, v_partial_dq_local_other, v_partial_dv_local_other,
    a_partial_dq_local_other, a_partial_dv_local_other, a_partial_da_local_other);

  BOOST_CHECK(v_partial_dq_local_other.isApprox(v_partial_dq_local));
  BOOST_CHECK(v_partial_dv_local_other.isApprox(a_partial_da_local));
  BOOST_CHECK(a_partial_dq_local_other.isApprox(a_partial_dq_local));
  BOOST_CHECK(a_partial_dv_local_other.isApprox(a_partial_dv_local));
  BOOST_CHECK(a_partial_da_local_other.isApprox(a_partial_da_local));
}

namespace
{
  struct FramePlacementDerivativeValues
  {
    pinocchio::SE3 placement;
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd spatial_acceleration;
    Eigen::VectorXd classical_acceleration;
  };

  FramePlacementDerivativeValues evaluateFrameKinematics(
    const pinocchio::Model & model,
    const Eigen::VectorXd & q,
    const Eigen::VectorXd & v,
    const pinocchio::FrameIndex frame_id,
    const pinocchio::ReferenceFrame reference_frame)
  {
    using namespace pinocchio;

    Data data(model);
    forwardKinematics(model, data, q, v, Eigen::VectorXd::Zero(model.nv));
    computeJointJacobians(model, data);
    updateFramePlacement(model, data, frame_id);

    FramePlacementDerivativeValues values;
    values.placement = data.oMf[frame_id];
    values.jacobian.setZero(6, model.nv);
    getFrameJacobian(model, data, frame_id, reference_frame, values.jacobian);
    values.spatial_acceleration =
      getFrameAcceleration(model, data, frame_id, reference_frame).toVector();
    values.classical_acceleration =
      getFrameClassicalAcceleration(model, data, frame_id, reference_frame).toVector();
    return values;
  }

  void finiteDifferenceFramePlacementDerivatives(
    const pinocchio::Model & model,
    const Eigen::VectorXd & q,
    const Eigen::VectorXd & v,
    const pinocchio::FrameIndex frame_id,
    const Eigen::MatrixXd & joint_placement_jacobians,
    const pinocchio::ReferenceFrame reference_frame,
    Eigen::MatrixXd & frame_placement_partial_dp,
    Eigen::MatrixXd & frame_jacobian_partial_dp,
    Eigen::MatrixXd & frame_acceleration_partial_dp,
    Eigen::MatrixXd & frame_classical_acceleration_partial_dp)
  {
    using namespace pinocchio;

    const double step = 1e-7;
    for (Eigen::Index parameter_id = 0; parameter_id < joint_placement_jacobians.cols();
         ++parameter_id)
    {
      Model model_plus(model), model_minus(model);
      for (JointIndex joint_id = 1; joint_id < (JointIndex)model.njoints; ++joint_id)
      {
        const Motion tangent(
          joint_placement_jacobians.col(parameter_id).segment<6>(6 * (Eigen::Index)joint_id));
        model_plus.jointPlacements[joint_id] =
          model.jointPlacements[joint_id] * exp6(step * tangent);
        model_minus.jointPlacements[joint_id] =
          model.jointPlacements[joint_id] * exp6(-step * tangent);
      }

      const FramePlacementDerivativeValues plus =
        evaluateFrameKinematics(model_plus, q, v, frame_id, reference_frame);
      const FramePlacementDerivativeValues minus =
        evaluateFrameKinematics(model_minus, q, v, frame_id, reference_frame);
      frame_placement_partial_dp.col(parameter_id) =
        log6(minus.placement.actInv(plus.placement)).toVector() / (2. * step);
      frame_jacobian_partial_dp.middleCols(parameter_id * model.nv, model.nv) =
        (plus.jacobian - minus.jacobian) / (2. * step);
      frame_acceleration_partial_dp.col(parameter_id) =
        (plus.spatial_acceleration - minus.spatial_acceleration) / (2. * step);
      frame_classical_acceleration_partial_dp.col(parameter_id) =
        (plus.classical_acceleration - minus.classical_acceleration) / (2. * step);
    }
  }

  void checkFramePlacementDerivatives(const bool using_mimic)
  {
    using namespace pinocchio;

    Model model;
    buildModels::manipulator(model, using_mimic);
    const Eigen::VectorXd q =
      integrate(model, neutral(model), 0.3 * Eigen::VectorXd::Random(model.nv));
    const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
    const FrameIndex frame_id = (FrameIndex)(model.nframes - 1);

    const Eigen::Index number_parameters = 3;
    Eigen::MatrixXd joint_placement_jacobians =
      0.2 * Eigen::MatrixXd::Random(6 * (Eigen::Index)model.njoints, number_parameters);
    joint_placement_jacobians.topRows(6).setZero();

    Data data(model), nominal_data(model);
    FramePlacementDerivativesWorkspaceTpl<double> workspace(model.nv, number_parameters + 2);
    const double * parameter_storage = workspace.parameter_storage.data();
    const double * jacobian_storage = workspace.jacobian_storage.data();
    const double * zero_acceleration_storage = workspace.zero_acceleration.data();

    const ReferenceFrame reference_frames[] = {LOCAL, LOCAL_WORLD_ALIGNED, WORLD};
    for (std::size_t reference_id = 0; reference_id < 3; ++reference_id)
    {
      const ReferenceFrame reference_frame = reference_frames[reference_id];
      Eigen::MatrixXd frame_placement_partial_dp(6, number_parameters);
      Eigen::MatrixXd frame_jacobian_partial_dp(6, model.nv * number_parameters);
      Eigen::MatrixXd frame_acceleration_partial_dp(6, number_parameters);
      Eigen::MatrixXd frame_classical_acceleration_partial_dp(6, number_parameters);
      computeFramePlacementDerivatives(
        model, data, q, v, frame_id, joint_placement_jacobians, reference_frame, workspace,
        frame_placement_partial_dp, frame_jacobian_partial_dp, frame_acceleration_partial_dp,
        frame_classical_acceleration_partial_dp);

      Eigen::MatrixXd frame_placement_partial_dp_fd(6, number_parameters);
      Eigen::MatrixXd frame_jacobian_partial_dp_fd(6, model.nv * number_parameters);
      Eigen::MatrixXd frame_acceleration_partial_dp_fd(6, number_parameters);
      Eigen::MatrixXd frame_classical_acceleration_partial_dp_fd(6, number_parameters);
      finiteDifferenceFramePlacementDerivatives(
        model, q, v, frame_id, joint_placement_jacobians, reference_frame,
        frame_placement_partial_dp_fd, frame_jacobian_partial_dp_fd,
        frame_acceleration_partial_dp_fd, frame_classical_acceleration_partial_dp_fd);

      BOOST_CHECK_SMALL(
        (frame_placement_partial_dp - frame_placement_partial_dp_fd).lpNorm<Eigen::Infinity>(),
        2e-5);
      BOOST_CHECK_SMALL(
        (frame_jacobian_partial_dp - frame_jacobian_partial_dp_fd).lpNorm<Eigen::Infinity>(), 2e-5);
      BOOST_CHECK_SMALL(
        (frame_acceleration_partial_dp - frame_acceleration_partial_dp_fd)
          .lpNorm<Eigen::Infinity>(),
        2e-5);
      BOOST_CHECK_SMALL(
        (frame_classical_acceleration_partial_dp - frame_classical_acceleration_partial_dp_fd)
          .lpNorm<Eigen::Infinity>(),
        2e-5);

      const Eigen::MatrixXd compact_joint_placement_jacobians =
        joint_placement_jacobians.bottomRows(6 * ((Eigen::Index)model.njoints - 1));
      Eigen::MatrixXd compact_frame_placement_partial_dp(6, number_parameters);
      Eigen::MatrixXd compact_frame_jacobian_partial_dp(6, model.nv * number_parameters);
      Eigen::MatrixXd compact_frame_acceleration_partial_dp(6, number_parameters);
      Eigen::MatrixXd compact_frame_classical_acceleration_partial_dp(6, number_parameters);
      computeFramePlacementDerivatives(
        model, data, q, v, frame_id, compact_joint_placement_jacobians, reference_frame, workspace,
        compact_frame_placement_partial_dp, compact_frame_jacobian_partial_dp,
        compact_frame_acceleration_partial_dp, compact_frame_classical_acceleration_partial_dp);
      BOOST_CHECK(compact_frame_placement_partial_dp.isApprox(frame_placement_partial_dp));
      BOOST_CHECK(compact_frame_jacobian_partial_dp.isApprox(frame_jacobian_partial_dp));
      BOOST_CHECK(compact_frame_acceleration_partial_dp.isApprox(frame_acceleration_partial_dp));
      BOOST_CHECK(compact_frame_classical_acceleration_partial_dp.isApprox(
        frame_classical_acceleration_partial_dp));

      forwardKinematics(model, nominal_data, q, v, Eigen::VectorXd::Zero(model.nv));
      computeJointJacobians(model, nominal_data);
      updateFramePlacement(model, nominal_data, frame_id);
      BOOST_CHECK(data.oMf[frame_id].isApprox(nominal_data.oMf[frame_id]));
      BOOST_CHECK(data.v[frame_id == 0 ? 0 : model.frames[frame_id].parentJoint].isApprox(
        nominal_data.v[frame_id == 0 ? 0 : model.frames[frame_id].parentJoint]));
    }

    BOOST_CHECK_EQUAL(parameter_storage, workspace.parameter_storage.data());
    BOOST_CHECK_EQUAL(jacobian_storage, workspace.jacobian_storage.data());
    BOOST_CHECK_EQUAL(zero_acceleration_storage, workspace.zero_acceleration.data());
  }
} // namespace

BOOST_AUTO_TEST_CASE(test_frame_placement_derivatives)
{
  checkFramePlacementDerivatives(false);
}

BOOST_AUTO_TEST_CASE(test_frame_placement_derivatives_mimic)
{
  checkFramePlacementDerivatives(true);
}

BOOST_AUTO_TEST_CASE(test_frame_placement_derivatives_ignore_joints_outside_support)
{
  using namespace pinocchio;

  Model model;
  buildModels::humanoidRandom(model);
  const JointIndex parent_joint = model.getJointId("rarm2_joint");
  const FrameIndex frame_id =
    model.addFrame(Frame("frame_placement_derivatives", parent_joint, 0, SE3::Random(), OP_FRAME));
  const Model::IndexVector & support = model.supports[parent_joint];

  JointIndex joint_outside_support = 0;
  for (JointIndex joint_id = 1; joint_id < (JointIndex)model.njoints; ++joint_id)
  {
    bool is_in_support = false;
    for (std::size_t support_id = 1; support_id < support.size(); ++support_id)
      is_in_support = is_in_support || support[support_id] == joint_id;
    if (!is_in_support)
    {
      joint_outside_support = joint_id;
      break;
    }
  }
  BOOST_REQUIRE(joint_outside_support != 0);

  Eigen::MatrixXd joint_placement_jacobians =
    Eigen::MatrixXd::Zero(6 * ((Eigen::Index)model.njoints - 1), 1);
  joint_placement_jacobians.middleRows<6>(6 * ((Eigen::Index)joint_outside_support - 1)) =
    Eigen::Matrix<double, 6, 1>::Random();
  const Eigen::VectorXd q =
    integrate(model, neutral(model), 0.3 * Eigen::VectorXd::Random(model.nv));
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  Data data(model);
  FramePlacementDerivativesWorkspace workspace(model.nv, 1);

  const ReferenceFrame reference_frames[] = {LOCAL, LOCAL_WORLD_ALIGNED, WORLD};
  for (std::size_t reference_id = 0; reference_id < 3; ++reference_id)
  {
    Eigen::MatrixXd frame_placement_partial_dp(6, 1);
    Eigen::MatrixXd frame_jacobian_partial_dp(6, model.nv);
    Eigen::MatrixXd frame_acceleration_partial_dp(6, 1);
    Eigen::MatrixXd frame_classical_acceleration_partial_dp(6, 1);
    computeFramePlacementDerivatives(
      model, data, q, v, frame_id, joint_placement_jacobians, reference_frames[reference_id],
      workspace, frame_placement_partial_dp, frame_jacobian_partial_dp,
      frame_acceleration_partial_dp, frame_classical_acceleration_partial_dp);

    BOOST_CHECK(frame_placement_partial_dp.isZero());
    BOOST_CHECK(frame_jacobian_partial_dp.isZero());
    BOOST_CHECK(frame_acceleration_partial_dp.isZero());
    BOOST_CHECK(frame_classical_acceleration_partial_dp.isZero());
  }
}

BOOST_AUTO_TEST_SUITE_END()
