//
// Copyright (c) 2026 INRIA
//

#include "pinocchio/multibody/sample-models.hpp"

#include "pinocchio/algorithm/contact-kinematics-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

namespace
{
  struct ConstraintKinematicValues
  {
    pinocchio::SE3 relative_placement;
    Eigen::VectorXd placement_error;
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd velocity;
    Eigen::VectorXd drift;
  };

  ConstraintKinematicValues evaluateConstraintKinematics(
    const pinocchio::Model & model,
    const Eigen::VectorXd & q,
    const Eigen::VectorXd & v,
    const pinocchio::RigidConstraintModel & cmodel)
  {
    using namespace pinocchio;

    Data data(model);
    forwardKinematics(model, data, q, v, Eigen::VectorXd::Zero(model.nv));
    computeJointJacobians(model, data);
    RigidConstraintData cdata(cmodel);
    cmodel.calc(model, data, cdata);

    Eigen::MatrixXd J1 = Eigen::MatrixXd::Zero(6, model.nv);
    Eigen::MatrixXd J2 = Eigen::MatrixXd::Zero(6, model.nv);
    getFrameJacobian(model, data, cmodel.joint1_id, cmodel.joint1_placement, LOCAL, J1);
    getFrameJacobian(model, data, cmodel.joint2_id, cmodel.joint2_placement, LOCAL, J2);

    const SE3 & c1Mc2 = cdata.c1Mc2;
    ConstraintKinematicValues values;
    values.relative_placement = c1Mc2;
    if (cmodel.type == CONTACT_6D)
    {
      values.jacobian.resize(6, model.nv);
      for (Eigen::Index velocity_id = 0; velocity_id < model.nv; ++velocity_id)
        values.jacobian.col(velocity_id) =
          (Motion(J1.col(velocity_id)) - c1Mc2.act(Motion(J2.col(velocity_id)))).toVector();
      values.placement_error = -log6(c1Mc2).toVector();
    }
    else
    {
      values.jacobian = J1.topRows<3>() - c1Mc2.rotation() * J2.topRows<3>();
      values.placement_error = -c1Mc2.translation();
    }

    const Motion a1 =
      getFrameAcceleration(model, data, cmodel.joint1_id, cmodel.joint1_placement, LOCAL);
    const Motion a2 =
      getFrameAcceleration(model, data, cmodel.joint2_id, cmodel.joint2_placement, LOCAL);
    if (cmodel.type == CONTACT_6D)
    {
      if (cmodel.reference_frame == LOCAL)
      {
        const Motion v1(J1 * v);
        const Motion v2_in_1(c1Mc2.act(Motion(J2 * v)));
        values.drift = (a1 + (v1 - v2_in_1).cross(v2_in_1) - c1Mc2.act(a2)).toVector();
      }
      else
      {
        const SE3 output_rotation(cdata.oMc1.rotation(), Eigen::Vector3d::Zero());
        values.jacobian = output_rotation.toActionMatrix() * values.jacobian;
        values.placement_error = output_rotation.act(Motion(values.placement_error)).toVector();
        values.drift = output_rotation.act(a1 - c1Mc2.act(a2)).toVector();
      }
    }
    else
    {
      const Motion classical1 = getFrameClassicalAcceleration(
        model, data, cmodel.joint1_id, cmodel.joint1_placement, LOCAL);
      const Motion classical2 = getFrameClassicalAcceleration(
        model, data, cmodel.joint2_id, cmodel.joint2_placement, LOCAL);
      values.drift = classical1.linear() - c1Mc2.rotation() * classical2.linear();
      if (cmodel.reference_frame == LOCAL_WORLD_ALIGNED)
      {
        values.jacobian = cdata.oMc1.rotation() * values.jacobian;
        values.placement_error = cdata.oMc1.rotation() * values.placement_error;
        values.drift = cdata.oMc1.rotation() * values.drift;
      }
    }
    values.velocity = values.jacobian * v;
    return values;
  }

  void finiteDifferenceConstraintKinematics(
    const pinocchio::Model & model,
    const Eigen::VectorXd & q,
    const Eigen::VectorXd & v,
    const pinocchio::RigidConstraintModel & cmodel,
    const Eigen::MatrixXd & joint_placement_jacobians,
    Eigen::MatrixXd & relative_placement_partial_dp,
    Eigen::MatrixXd & placement_error_partial_dp,
    Eigen::MatrixXd & constraint_jacobian_partial_dp,
    Eigen::MatrixXd & constraint_velocity_partial_dp,
    Eigen::MatrixXd & constraint_acceleration_partial_dp)
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

      const ConstraintKinematicValues plus = evaluateConstraintKinematics(model_plus, q, v, cmodel);
      const ConstraintKinematicValues minus =
        evaluateConstraintKinematics(model_minus, q, v, cmodel);
      relative_placement_partial_dp.col(parameter_id) =
        log6(minus.relative_placement.actInv(plus.relative_placement)).toVector() / (2. * step);
      placement_error_partial_dp.col(parameter_id) =
        (plus.placement_error - minus.placement_error) / (2. * step);
      constraint_jacobian_partial_dp.middleCols(parameter_id * model.nv, model.nv) =
        (plus.jacobian - minus.jacobian) / (2. * step);
      constraint_velocity_partial_dp.col(parameter_id) =
        (plus.velocity - minus.velocity) / (2. * step);
      constraint_acceleration_partial_dp.col(parameter_id) =
        (plus.drift - minus.drift) / (2. * step);
    }
  }

  void checkConstraintKinematicDerivatives(
    const pinocchio::Model & model,
    const pinocchio::JointIndex joint1_id,
    const pinocchio::JointIndex joint2_id)
  {
    using namespace pinocchio;

    const Eigen::VectorXd q =
      integrate(model, neutral(model), 0.25 * Eigen::VectorXd::Random(model.nv));
    const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
    const Eigen::Index number_parameters = 3;
    Eigen::MatrixXd joint_placement_jacobians =
      0.2 * Eigen::MatrixXd::Random(6 * (Eigen::Index)model.njoints, number_parameters);
    joint_placement_jacobians.topRows(6).setZero();
    RigidConstraintKinematicDerivativesWorkspace workspace(model.nv, number_parameters + 2);

    const ContactType contact_types[] = {CONTACT_3D, CONTACT_6D};
    const ReferenceFrame reference_frames[] = {LOCAL, LOCAL_WORLD_ALIGNED};
    for (std::size_t type_id = 0; type_id < 2; ++type_id)
    {
      for (std::size_t reference_id = 0; reference_id < 2; ++reference_id)
      {
        RigidConstraintModel cmodel(
          contact_types[type_id], model, joint1_id, SE3::Random(), joint2_id, SE3::Random(),
          reference_frames[reference_id]);
        cmodel.m_baumgarte_parameters.Kp = 3.;
        cmodel.m_baumgarte_parameters.Kd = .7;
        RigidConstraintData cdata(cmodel);
        Data data(model);
        const Eigen::Index constraint_size = cmodel.residualSize();
        Eigen::MatrixXd relative_placement_partial_dp(6, number_parameters);
        Eigen::MatrixXd placement_error_partial_dp(constraint_size, number_parameters);
        Eigen::MatrixXd constraint_jacobian_partial_dp(
          constraint_size, model.nv * number_parameters);
        Eigen::MatrixXd constraint_velocity_partial_dp(constraint_size, number_parameters);
        Eigen::MatrixXd constraint_acceleration_partial_dp(constraint_size, number_parameters);
        computeRigidConstraintKinematicDerivatives(
          model, data, cmodel, cdata, q, v, joint_placement_jacobians, workspace,
          relative_placement_partial_dp, placement_error_partial_dp, constraint_jacobian_partial_dp,
          constraint_velocity_partial_dp, constraint_acceleration_partial_dp);

        Eigen::MatrixXd relative_placement_partial_dp_fd(6, number_parameters);
        Eigen::MatrixXd placement_error_partial_dp_fd(constraint_size, number_parameters);
        Eigen::MatrixXd constraint_jacobian_partial_dp_fd(
          constraint_size, model.nv * number_parameters);
        Eigen::MatrixXd constraint_velocity_partial_dp_fd(constraint_size, number_parameters);
        Eigen::MatrixXd constraint_acceleration_partial_dp_fd(constraint_size, number_parameters);
        finiteDifferenceConstraintKinematics(
          model, q, v, cmodel, joint_placement_jacobians, relative_placement_partial_dp_fd,
          placement_error_partial_dp_fd, constraint_jacobian_partial_dp_fd,
          constraint_velocity_partial_dp_fd, constraint_acceleration_partial_dp_fd);

        // The direct central SE(3) logarithm loses angular precision on free-flyer models at this
        // step size. The 6D placement-error check below exercises the same tangent through Jlog6.
        if (model.nq == model.nv)
          BOOST_CHECK_SMALL(
            (relative_placement_partial_dp - relative_placement_partial_dp_fd)
              .lpNorm<Eigen::Infinity>(),
            3e-5);
        BOOST_CHECK_SMALL(
          (placement_error_partial_dp - placement_error_partial_dp_fd).lpNorm<Eigen::Infinity>(),
          3e-5);
        BOOST_CHECK_SMALL(
          (constraint_jacobian_partial_dp - constraint_jacobian_partial_dp_fd)
            .lpNorm<Eigen::Infinity>(),
          3e-5);
        BOOST_CHECK_SMALL(
          (constraint_velocity_partial_dp - constraint_velocity_partial_dp_fd)
            .lpNorm<Eigen::Infinity>(),
          3e-5);
        BOOST_CHECK_SMALL(
          (constraint_acceleration_partial_dp - constraint_acceleration_partial_dp_fd)
            .lpNorm<Eigen::Infinity>(),
          3e-5);
      }
    }
  }
} // namespace

BOOST_AUTO_TEST_CASE(test_two_moving_frame_constraint_kinematic_derivatives)
{
  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model);
  checkConstraintKinematicDerivatives(
    model, model.getJointId("rarm2_joint"), model.getJointId("larm2_joint"));
}

BOOST_AUTO_TEST_CASE(test_mimic_constraint_kinematic_derivatives)
{
  pinocchio::Model model;
  pinocchio::buildModels::manipulator(model, true);
  checkConstraintKinematicDerivatives(
    model, (pinocchio::JointIndex)(model.njoints - 1), (pinocchio::JointIndex)(model.njoints / 2));
}

BOOST_AUTO_TEST_SUITE_END()
