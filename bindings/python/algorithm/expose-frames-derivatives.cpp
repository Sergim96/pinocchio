//
// Copyright (c) 2020 INRIA
//

#include "pinocchio/bindings/python/fwd.hpp"
#include <boost/python/tuple.hpp>

#include "pinocchio/bindings/python/algorithm/algorithms.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"

#include "pinocchio/bindings/python/utils/model-checker.hpp"

namespace pinocchio
{
  namespace python
  {
    namespace bp = boost::python;

    typedef FramePlacementDerivativesWorkspaceTpl<context::Scalar, context::Options>
      FramePlacementDerivativesWorkspace;

    bp::tuple computeFramePlacementDerivatives_proxy(
      const context::Model & model,
      context::Data & data,
      const context::VectorXs & q,
      const context::VectorXs & v,
      const context::Model::FrameIndex frame_id,
      const context::MatrixXs & joint_placement_jacobians,
      const ReferenceFrame reference_frame,
      FramePlacementDerivativesWorkspace & workspace)
    {
      typedef context::Data::Tensor3x Tensor3x;

      const Eigen::Index number_parameters = joint_placement_jacobians.cols();
      context::MatrixXs frame_placement_partial_dp(6, number_parameters);
      context::MatrixXs frame_jacobian_partial_dp_matrix(6, model.nv * number_parameters);
      context::MatrixXs frame_acceleration_partial_dp(6, number_parameters);
      context::MatrixXs frame_classical_acceleration_partial_dp(6, number_parameters);

      computeFramePlacementDerivatives(
        model, data, q, v, frame_id, joint_placement_jacobians, reference_frame, workspace,
        frame_placement_partial_dp, frame_jacobian_partial_dp_matrix, frame_acceleration_partial_dp,
        frame_classical_acceleration_partial_dp);

      // eigenpy exposes tensors as C-contiguous NumPy arrays. Reorder the native
      // parameter-major matrix into the returned (6, nv, number_parameters) array.
      Tensor3x frame_jacobian_partial_dp(6, model.nv, number_parameters);
      for (Eigen::Index row = 0; row < 6; ++row)
      {
        for (Eigen::Index velocity_id = 0; velocity_id < model.nv; ++velocity_id)
        {
          for (Eigen::Index parameter_id = 0; parameter_id < number_parameters; ++parameter_id)
          {
            const Eigen::Index tensor_offset =
              (row * model.nv + velocity_id) * number_parameters + parameter_id;
            frame_jacobian_partial_dp.data()[tensor_offset] =
              frame_jacobian_partial_dp_matrix(row, parameter_id * model.nv + velocity_id);
          }
        }
      }
      return bp::make_tuple(
        frame_placement_partial_dp, frame_jacobian_partial_dp, frame_acceleration_partial_dp,
        frame_classical_acceleration_partial_dp);
    }

    bp::tuple computeFramePlacementDerivatives_proxy(
      const context::Model & model,
      context::Data & data,
      const context::VectorXs & q,
      const context::VectorXs & v,
      const context::Model::FrameIndex frame_id,
      const context::MatrixXs & joint_placement_jacobians,
      const ReferenceFrame reference_frame)
    {
      FramePlacementDerivativesWorkspace workspace(model.nv, joint_placement_jacobians.cols());
      return computeFramePlacementDerivatives_proxy(
        model, data, q, v, frame_id, joint_placement_jacobians, reference_frame, workspace);
    }

    bp::tuple getFrameVelocityDerivatives_proxy1(
      const context::Model & model,
      context::Data & data,
      const context::Model::FrameIndex frame_id,
      ReferenceFrame rf)
    {
      typedef context::Data::Matrix6x Matrix6x;

      Matrix6x partial_dq(Matrix6x::Zero(6, model.nv));
      Matrix6x partial_dv(Matrix6x::Zero(6, model.nv));

      getFrameVelocityDerivatives(model, data, frame_id, rf, partial_dq, partial_dv);

      return bp::make_tuple(partial_dq, partial_dv);
    }

    bp::tuple getFrameVelocityDerivatives_proxy2(
      const context::Model & model,
      context::Data & data,
      const context::Model::JointIndex joint_id,
      const context::SE3 & placement,
      ReferenceFrame rf)
    {
      typedef context::Data::Matrix6x Matrix6x;

      Matrix6x partial_dq(Matrix6x::Zero(6, model.nv));
      Matrix6x partial_dv(Matrix6x::Zero(6, model.nv));

      getFrameVelocityDerivatives(model, data, joint_id, placement, rf, partial_dq, partial_dv);

      return bp::make_tuple(partial_dq, partial_dv);
    }

    bp::tuple getFrameAccelerationDerivatives_proxy1(
      const context::Model & model,
      context::Data & data,
      const context::Model::FrameIndex frame_id,
      ReferenceFrame rf)
    {
      typedef context::Data::Matrix6x Matrix6x;

      Matrix6x v_partial_dq(Matrix6x::Zero(6, model.nv));
      Matrix6x a_partial_dq(Matrix6x::Zero(6, model.nv));
      Matrix6x a_partial_dv(Matrix6x::Zero(6, model.nv));
      Matrix6x a_partial_da(Matrix6x::Zero(6, model.nv));

      getFrameAccelerationDerivatives(
        model, data, frame_id, rf, v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da);

      return bp::make_tuple(v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da);
    }

    bp::tuple getFrameAccelerationDerivatives_proxy2(
      const context::Model & model,
      context::Data & data,
      const context::Model::JointIndex joint_id,
      const context::SE3 & placement,
      ReferenceFrame rf)
    {
      typedef context::Data::Matrix6x Matrix6x;

      Matrix6x v_partial_dq(Matrix6x::Zero(6, model.nv));
      Matrix6x a_partial_dq(Matrix6x::Zero(6, model.nv));
      Matrix6x a_partial_dv(Matrix6x::Zero(6, model.nv));
      Matrix6x a_partial_da(Matrix6x::Zero(6, model.nv));

      getFrameAccelerationDerivatives(
        model, data, joint_id, placement, rf, v_partial_dq, a_partial_dq, a_partial_dv,
        a_partial_da);

      return bp::make_tuple(v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da);
    }

    void exposeFramesDerivatives()
    {
      using namespace Eigen;

      bp::class_<FramePlacementDerivativesWorkspace>(
        "FramePlacementDerivativesWorkspace",
        "Reusable native workspace for frame joint-placement derivatives.", bp::init<>())
        .def(bp::init<Eigen::Index, Eigen::Index>((bp::args("nv", "parameter_capacity"))))
        .def(
          "resize", &FramePlacementDerivativesWorkspace::resize,
          bp::args("self", "nv", "parameter_capacity"))
        .def_readonly("nv", &FramePlacementDerivativesWorkspace::nv)
        .def_readonly(
          "parameter_capacity", &FramePlacementDerivativesWorkspace::parameter_capacity);

      bp::def(
        "computeFramePlacementDerivatives",
        static_cast<bp::tuple (*)(
          const context::Model &, context::Data &, const context::VectorXs &,
          const context::VectorXs &, const context::Model::FrameIndex, const context::MatrixXs &,
          const ReferenceFrame)>(&computeFramePlacementDerivatives_proxy),
        (bp::arg("model"), bp::arg("data"), bp::arg("q"), bp::arg("v"), bp::arg("frame_id"),
         bp::arg("joint_placement_jacobians"), bp::arg("reference_frame") = LOCAL),
        "Computes frame placement, Jacobian, spatial drift and classical drift derivatives with "
        "respect to "
        "right-trivialized joint-placement perturbations.\n\n"
        "The placement Jacobian has 6 * model.njoints or 6 * (model.njoints - 1) rows and np "
        "columns.\n"
        "Returns arrays with shapes (6, np), (6, model.nv, np), (6, np), and (6, np).");

      bp::def(
        "computeFramePlacementDerivatives",
        static_cast<bp::tuple (*)(
          const context::Model &, context::Data &, const context::VectorXs &,
          const context::VectorXs &, const context::Model::FrameIndex, const context::MatrixXs &,
          const ReferenceFrame, FramePlacementDerivativesWorkspace &)>(
          &computeFramePlacementDerivatives_proxy),
        bp::args(
          "model", "data", "q", "v", "frame_id", "joint_placement_jacobians", "reference_frame",
          "workspace"),
        "Computes frame joint-placement derivatives using a reusable native workspace.");

      bp::def(
        "getFrameVelocityDerivatives", getFrameVelocityDerivatives_proxy1,
        bp::args("model", "data", "frame_id", "reference_frame"),
        "Computes the partial derivatives of the spatial velocity of a given frame with respect "
        "to\n"
        "the joint configuration and velocity and returns them as a tuple.\n"
        "The partial derivatives can be either expressed in the LOCAL frame of the joint, in the "
        "LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of "
        "reference_frame.\n"
        "You must first call computeForwardKinematicsDerivatives before calling this function.\n\n"
        "Parameters:\n"
        "\tmodel: model of the kinematic tree\n"
        "\tdata: data related to the model\n"
        "\tframe_id: index of the frame\n"
        "\treference_frame: reference frame in which the resulting derivatives are expressed\n",
        mimic_not_supported_function<>(0));

      bp::def(
        "getFrameVelocityDerivatives", getFrameVelocityDerivatives_proxy2,
        bp::args("model", "data", "joint_id", "placement", "reference_frame"),
        "Computes the partial derivatives of the spatial velocity of a frame given by its relative "
        "placement, with respect to\n"
        "the joint configuration and velocity and returns them as a tuple.\n"
        "The partial derivatives can be either expressed in the LOCAL frame of the joint, in the "
        "LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of "
        "reference_frame.\n"
        "You must first call computeForwardKinematicsDerivatives before calling this function.\n\n"
        "Parameters:\n"
        "\tmodel: model of the kinematic tree\n"
        "\tdata: data related to the model\n"
        "\tjoint_id: index of the joint\n"
        "\tplacement: placement of the Frame w.r.t. the joint frame.\n"
        "\treference_frame: reference frame in which the resulting derivatives are expressed\n",
        mimic_not_supported_function<>(0));

      bp::def(
        "getFrameAccelerationDerivatives", getFrameAccelerationDerivatives_proxy1,
        bp::args("model", "data", "frame_id", "reference_frame"),
        "Computes the partial derivatives of the spatial acceleration of a given frame with "
        "respect to\n"
        "the joint configuration, velocity and acceleration and returns them as a tuple.\n"
        "The partial derivatives can be either expressed in the LOCAL frame of the joint, in the "
        "LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of "
        "reference_frame.\n"
        "You must first call computeForwardKinematicsDerivatives before calling this function.\n\n"
        "Parameters:\n"
        "\tmodel: model of the kinematic tree\n"
        "\tdata: data related to the model\n"
        "\tframe_id: index of the frame\n"
        "\treference_frame: reference frame in which the resulting derivatives are expressed\n",
        mimic_not_supported_function<>(0));

      bp::def(
        "getFrameAccelerationDerivatives", getFrameAccelerationDerivatives_proxy2,
        bp::args("model", "data", "joint_id", "placement", "reference_frame"),
        "Computes the partial derivatives of the spatial acceleration of a frame given by its "
        "relative placement, with respect to\n"
        "the joint configuration, velocity and acceleration and returns them as a tuple.\n"
        "The partial derivatives can be either expressed in the LOCAL frame of the joint, in the "
        "LOCAL_WORLD_ALIGNED frame or in the WORLD coordinate frame depending on the value of "
        "reference_frame.\n"
        "You must first call computeForwardKinematicsDerivatives before calling this function.\n\n"
        "Parameters:\n"
        "\tmodel: model of the kinematic tree\n"
        "\tdata: data related to the model\n"
        "\tjoint_id: index of the joint\n"
        "\tplacement: placement of the Frame w.r.t. the joint frame.\n"
        "\treference_frame: reference frame in which the resulting derivatives are expressed\n",
        mimic_not_supported_function<>(0));
    }

  } // namespace python
} // namespace pinocchio
