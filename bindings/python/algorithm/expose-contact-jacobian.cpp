//
// Copyright (c) 2021-2025 INRIA
//

#include "pinocchio/bindings/python/algorithm/algorithms.hpp"
#include "pinocchio/constraints.hpp"
#include "pinocchio/algorithm/contact-kinematics-derivatives.hpp"

#include "pinocchio/bindings/python/utils/std-vector.hpp"
#include "pinocchio/bindings/python/utils/model-checker.hpp"

namespace bp = boost::python;

namespace pinocchio
{
  namespace python
  {

    typedef RigidConstraintKinematicDerivativesWorkspaceTpl<context::Scalar, context::Options>
      RigidConstraintKinematicDerivativesWorkspace;

    static bp::tuple computeRigidConstraintKinematicDerivatives_proxy(
      const context::Model & model,
      context::Data & data,
      const context::RigidConstraintModel & constraint_model,
      context::RigidConstraintData & constraint_data,
      const context::VectorXs & q,
      const context::VectorXs & v,
      const context::MatrixXs & joint_placement_jacobians,
      RigidConstraintKinematicDerivativesWorkspace & workspace)
    {
      typedef context::Data::Tensor3x Tensor3x;

      const Eigen::Index number_parameters = joint_placement_jacobians.cols();
      const Eigen::Index constraint_size = constraint_model.residualSize();
      context::MatrixXs relative_placement_partial_dp(6, number_parameters);
      context::MatrixXs placement_error_partial_dp(constraint_size, number_parameters);
      context::MatrixXs constraint_jacobian_partial_dp_matrix(
        constraint_size, model.nv * number_parameters);
      context::MatrixXs constraint_velocity_partial_dp(constraint_size, number_parameters);
      context::MatrixXs constraint_acceleration_partial_dp(constraint_size, number_parameters);

      computeRigidConstraintKinematicDerivatives(
        model, data, constraint_model, constraint_data, q, v, joint_placement_jacobians, workspace,
        relative_placement_partial_dp, placement_error_partial_dp,
        constraint_jacobian_partial_dp_matrix, constraint_velocity_partial_dp,
        constraint_acceleration_partial_dp);

      Tensor3x constraint_jacobian_partial_dp(constraint_size, model.nv, number_parameters);
      for (Eigen::Index row = 0; row < constraint_size; ++row)
      {
        for (Eigen::Index velocity_id = 0; velocity_id < model.nv; ++velocity_id)
        {
          for (Eigen::Index parameter_id = 0; parameter_id < number_parameters; ++parameter_id)
          {
            const Eigen::Index tensor_offset =
              (row * model.nv + velocity_id) * number_parameters + parameter_id;
            constraint_jacobian_partial_dp.data()[tensor_offset] =
              constraint_jacobian_partial_dp_matrix(row, parameter_id * model.nv + velocity_id);
          }
        }
      }

      return bp::make_tuple(
        relative_placement_partial_dp, placement_error_partial_dp, constraint_jacobian_partial_dp,
        constraint_velocity_partial_dp, constraint_acceleration_partial_dp);
    }

    static bp::tuple computeRigidConstraintKinematicDerivatives_proxy(
      const context::Model & model,
      context::Data & data,
      const context::RigidConstraintModel & constraint_model,
      context::RigidConstraintData & constraint_data,
      const context::VectorXs & q,
      const context::VectorXs & v,
      const context::MatrixXs & joint_placement_jacobians)
    {
      RigidConstraintKinematicDerivativesWorkspace workspace(
        model.nv, joint_placement_jacobians.cols());
      return computeRigidConstraintKinematicDerivatives_proxy(
        model, data, constraint_model, constraint_data, q, v, joint_placement_jacobians, workspace);
    }

    template<typename ConstraintModel, typename ConstraintData>
    static context::MatrixXs getConstraintJacobian_proxy(
      const context::Model & model,
      const context::Data & data,
      const ConstraintModel & constraint_model,
      const ConstraintData & constraint_data)
    {
      context::MatrixXs J(constraint_model.residualSize(), model.nv);
      J.setZero();
      getConstraintJacobian(model, data, constraint_model, constraint_data, J);
      return J;
    }

    template<typename ConstraintModelVector, typename ConstraintDataVector>
    static context::MatrixXs getConstraintsJacobian_proxy(
      const context::Model & model,
      const context::Data & data,
      const ConstraintModelVector & constraint_models,
      const ConstraintDataVector & constraint_datas)
    {
      const Eigen::Index constraint_residual_size =
        getTotalConstraintResidualSize(constraint_models);
      context::MatrixXs J(constraint_residual_size, model.nv);
      J.setZero();
      getConstraintsJacobian(model, data, constraint_models, constraint_datas, J);
      return J;
    }

    void exposeContactJacobian()
    {
      bp::class_<RigidConstraintKinematicDerivativesWorkspace>(
        "RigidConstraintKinematicDerivativesWorkspace",
        "Reusable native workspace for rigid-constraint joint-placement derivatives.", bp::init<>())
        .def(bp::init<Eigen::Index, Eigen::Index>((bp::args("nv", "parameter_capacity"))))
        .def(
          "resize", &RigidConstraintKinematicDerivativesWorkspace::resize,
          bp::args("self", "nv", "parameter_capacity"))
        .def_readonly("nv", &RigidConstraintKinematicDerivativesWorkspace::nv)
        .def_readonly(
          "parameter_capacity", &RigidConstraintKinematicDerivativesWorkspace::parameter_capacity);

      bp::def(
        "computeRigidConstraintKinematicDerivatives",
        static_cast<bp::tuple (*)(
          const context::Model &, context::Data &, const context::RigidConstraintModel &,
          context::RigidConstraintData &, const context::VectorXs &, const context::VectorXs &,
          const context::MatrixXs &)>(&computeRigidConstraintKinematicDerivatives_proxy),
        bp::args(
          "model", "data", "constraint_model", "constraint_data", "q", "v",
          "joint_placement_jacobians"),
        "Computes relative placement, placement-error, Jacobian, velocity and drift derivatives "
        "for a two-frame rigid constraint.\n\n"
        "Returns arrays with shapes (6, np), (nc, np), (nc, model.nv, np), (nc, np), "
        "and (nc, np).");
      bp::def(
        "computeRigidConstraintKinematicDerivatives",
        static_cast<bp::tuple (*)(
          const context::Model &, context::Data &, const context::RigidConstraintModel &,
          context::RigidConstraintData &, const context::VectorXs &, const context::VectorXs &,
          const context::MatrixXs &, RigidConstraintKinematicDerivativesWorkspace &)>(
          &computeRigidConstraintKinematicDerivatives_proxy),
        bp::args(
          "model", "data", "constraint_model", "constraint_data", "q", "v",
          "joint_placement_jacobians", "workspace"),
        "Computes rigid-constraint joint-placement derivatives using a reusable workspace.");

      bp::def(
        "getConstraintJacobian",
        getConstraintJacobian_proxy<context::RigidConstraintModel, context::RigidConstraintData>,
        bp::args("model", "data", "constraint_model", "constraint_data"),
        "Computes the kinematic Jacobian associatied with a given constraint model.",
        mimic_not_supported_function<>(0));
      bp::def(
        "getConstraintsJacobian",
        getConstraintsJacobian_proxy<
          context::RigidConstraintModelVector, context::RigidConstraintDataVector>,
        bp::args("model", "data", "constraint_models", "constraint_datas"),
        "Computes the kinematic Jacobian associatied with a given set of constraint models.",
        mimic_not_supported_function<>(0));

      bp::def(
        "getConstraintJacobian",
        getConstraintJacobian_proxy<context::ConstraintModel, context::ConstraintData>,
        bp::args("model", "data", "constraint_model", "constraint_data"),
        "Computes the kinematic Jacobian associatied with a given constraint model.",
        mimic_not_supported_function<>(0));
      bp::def(
        "getConstraintsJacobian",
        getConstraintsJacobian_proxy<context::ConstraintModelVector, context::ConstraintDataVector>,
        bp::args("model", "data", "constraint_models", "constraint_datas"),
        "Computes the kinematic Jacobian associatied with a given set of constraint models.",
        mimic_not_supported_function<>(0));
    }
  } // namespace python
} // namespace pinocchio
