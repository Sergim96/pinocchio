//
// Copyright (c) 2018-2021 CNRS INRIA
//

#include "pinocchio/bindings/python/algorithm/algorithms.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/bindings/python/utils/eigen.hpp"
#include "pinocchio/bindings/python/utils/model-checker.hpp"

namespace pinocchio
{
  namespace python
  {

    namespace bp = boost::python;
    typedef std::vector<context::Force> ForceAlignedVector;
    typedef RNEAPlacementDerivativesWorkspaceTpl<context::Scalar, context::Options>
      RNEAPlacementDerivativesWorkspace;

    context::Data::MatrixXs computeRNEAPlacementDerivatives(
      const context::Model & model,
      context::Data & data,
      const context::VectorXs & q,
      const context::VectorXs & v,
      const context::VectorXs & a,
      const context::MatrixXs & joint_placement_jacobians,
      RNEAPlacementDerivativesWorkspace & workspace)
    {
      context::Data::MatrixXs res(model.nv, joint_placement_jacobians.cols());
      pinocchio::computeRNEAPlacementDerivatives(
        model, data, q, v, a, joint_placement_jacobians, workspace, res);
      return res;
    }

    context::Data::MatrixXs computeRNEAPlacementDerivatives(
      const context::Model & model,
      context::Data & data,
      const context::VectorXs & q,
      const context::VectorXs & v,
      const context::VectorXs & a,
      const context::MatrixXs & joint_placement_jacobians)
    {
      RNEAPlacementDerivativesWorkspace workspace(
        (Eigen::Index)model.njoints, joint_placement_jacobians.cols());
      return computeRNEAPlacementDerivatives(
        model, data, q, v, a, joint_placement_jacobians, workspace);
    }

    context::Data::MatrixXs computeGeneralizedGravityDerivatives(
      const context::Model & model, context::Data & data, const context::VectorXs & q)
    {
      context::Data::MatrixXs res(model.nv, model.nv);
      res.setZero();
      pinocchio::computeGeneralizedGravityDerivatives(model, data, q, res);
      return res;
    }

    context::Data::MatrixXs computeStaticTorqueDerivatives(
      const context::Model & model,
      context::Data & data,
      const context::VectorXs & q,
      const ForceAlignedVector & fext)
    {
      context::Data::MatrixXs res(model.nv, model.nv);
      res.setZero();
      pinocchio::computeStaticTorqueDerivatives(model, data, q, fext, res);
      return res;
    }

    bp::tuple computeRNEADerivatives(
      const context::Model & model,
      context::Data & data,
      const context::VectorXs & q,
      const context::VectorXs & v,
      const context::VectorXs & a)
    {
      pinocchio::computeRNEADerivatives(model, data, q, v, a);
      make_symmetric(data.M);
      return bp::make_tuple(make_ref(data.dtau_dq), make_ref(data.dtau_dv), make_ref(data.M));
    }

    bp::tuple computeRNEADerivatives_fext(
      const context::Model & model,
      context::Data & data,
      const context::VectorXs & q,
      const context::VectorXs & v,
      const context::VectorXs & a,
      const ForceAlignedVector & fext)
    {
      pinocchio::computeRNEADerivatives(model, data, q, v, a, fext);
      make_symmetric(data.M);
      return bp::make_tuple(make_ref(data.dtau_dq), make_ref(data.dtau_dv), make_ref(data.M));
    }

    void exposeRNEADerivatives()
    {
      bp::class_<RNEAPlacementDerivativesWorkspace>(
        "RNEAPlacementDerivativesWorkspace",
        "Reusable native workspace for RNEA joint-placement derivatives.", bp::init<>())
        .def(bp::init<Eigen::Index, Eigen::Index>((bp::args("njoints", "parameter_capacity"))))
        .def(
          "resize", &RNEAPlacementDerivativesWorkspace::resize,
          bp::args("self", "njoints", "parameter_capacity"))
        .def_readonly("njoints", &RNEAPlacementDerivativesWorkspace::njoints)
        .def_readonly("parameter_capacity", &RNEAPlacementDerivativesWorkspace::parameter_capacity);

      bp::def(
        "computeRNEAPlacementDerivatives",
        static_cast<context::Data::MatrixXs (*)(
          const context::Model &, context::Data &, const context::VectorXs &,
          const context::VectorXs &, const context::VectorXs &, const context::MatrixXs &)>(
          &computeRNEAPlacementDerivatives),
        bp::args("model", "data", "q", "v", "a", "joint_placement_jacobians"),
        "Computes RNEA derivatives with respect to right-trivialized joint-placement "
        "perturbations.\n\n"
        "The placement Jacobian is a (6 * model.njoints) by np stacked matrix.\n"
        "Returns a model.nv by np generalized torque derivative matrix.");

      bp::def(
        "computeRNEAPlacementDerivatives",
        static_cast<context::Data::MatrixXs (*)(
          const context::Model &, context::Data &, const context::VectorXs &,
          const context::VectorXs &, const context::VectorXs &, const context::MatrixXs &,
          RNEAPlacementDerivativesWorkspace &)>(&computeRNEAPlacementDerivatives),
        bp::args("model", "data", "q", "v", "a", "joint_placement_jacobians", "workspace"),
        "Computes RNEA joint-placement derivatives using a reusable native workspace.");

      bp::def(
        "computeGeneralizedGravityDerivatives", computeGeneralizedGravityDerivatives,
        bp::args("model", "data", "q"),
        "Computes the partial derivative of the generalized gravity contribution\n"
        "with respect to the joint configuration.\n\n"
        "Parameters:\n"
        "\tmodel: model of the kinematic tree\n"
        "\tdata: data related to the model\n"
        "\tq: the joint configuration vector (size model.nq)\n"
        "Returns: dtau_statique_dq\n",
        mimic_not_supported_function<>(0));

      bp::def(
        "computeStaticTorqueDerivatives", computeStaticTorqueDerivatives,
        bp::args("model", "data", "q", "fext"),
        "Computes the partial derivative of the generalized gravity and external forces "
        "contributions (a.k.a static torque vector)\n"
        "with respect to the joint configuration.\n\n"
        "Parameters:\n"
        "\tmodel: model of the kinematic tree\n"
        "\tdata: data related to the model\n"
        "\tq: the joint configuration vector (size model.nq)\n"
        "\tfext: list of external forces expressed in the local frame of the joints (size "
        "model.njoints)\n"
        "Returns: dtau_statique_dq\n",
        mimic_not_supported_function<>(0));

      bp::def(
        "computeRNEADerivatives", computeRNEADerivatives, bp::args("model", "data", "q", "v", "a"),
        "Computes the RNEA partial derivatives, store the result in data.dtau_dq, data.dtau_dv and "
        "data.M (aka dtau_da)\n"
        "which correspond to the partial derivatives of the torque output with respect to the "
        "joint configuration,\n"
        "velocity and acceleration vectors.\n\n"
        "Parameters:\n"
        "\tmodel: model of the kinematic tree\n"
        "\tdata: data related to the model\n"
        "\tq: the joint configuration vector (size model.nq)\n"
        "\tv: the joint velocity vector (size model.nv)\n"
        "\ta: the joint acceleration vector (size model.nv)\n\n"
        "Returns: (dtau_dq, dtau_dv, dtau_da)\n",
        mimic_not_supported_function<>(0));

      bp::def(
        "computeRNEADerivatives", computeRNEADerivatives_fext,
        bp::args("model", "data", "q", "v", "a", "fext"),
        "Computes the RNEA partial derivatives with external contact foces,\n"
        "store the result in data.dtau_dq, data.dtau_dv and data.M (aka dtau_da)\n"
        "which correspond to the partial derivatives of the torque output with respect to the "
        "joint configuration,\n"
        "velocity and acceleration vectors.\n\n"
        "Parameters:\n"
        "\tmodel: model of the kinematic tree\n"
        "\tdata: data related to the model\n"
        "\tq: the joint configuration vector (size model.nq)\n"
        "\tv: the joint velocity vector (size model.nv)\n"
        "\ta: the joint acceleration vector (size model.nv)\n"
        "\tfext: list of external forces expressed in the local frame of the joints (size "
        "model.njoints)\n\n"
        "Returns: (dtau_dq, dtau_dv, dtau_da)\n",
        mimic_not_supported_function<>(0));
    }

  } // namespace python
} // namespace pinocchio
