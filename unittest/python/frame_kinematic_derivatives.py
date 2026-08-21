import unittest

import numpy as np
import pinocchio as pin
from pinocchio.derivative.frame_kinematic import (
    compute_frame_placement_derivatives,
    compute_frame_placement_derivatives_full,
)
from test_case import PinocchioTestCase as TestCase


class TestFrameKinematicDerivatives(TestCase):
    finite_difference_step = 1e-7

    @staticmethod
    def make_state(model, seed):
        random = np.random.default_rng(seed)
        q = pin.integrate(
            model, pin.neutral(model), 0.3 * random.standard_normal(model.nv)
        )
        v = random.standard_normal(model.nv)
        return q, v

    @staticmethod
    def evaluate(model, q, v, frame_id, reference_frame):
        data = model.createData()
        pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))
        pin.computeJointJacobians(model, data)
        pin.updateFramePlacements(model, data)
        return (
            data.oMf[frame_id].copy(),
            pin.getFrameJacobian(model, data, frame_id, reference_frame).copy(),
            pin.getFrameAcceleration(
                model, data, frame_id, reference_frame
            ).vector.copy(),
            pin.getFrameClassicalAcceleration(
                model, data, frame_id, reference_frame
            ).vector.copy(),
        )

    @classmethod
    def finite_difference(cls, model, q, v, frame_id, reference_frame):
        number_parameters = 6 * (model.njoints - 1)
        placement = np.zeros((6, number_parameters))
        jacobian = np.zeros((6, model.nv, number_parameters))
        spatial_drift = np.zeros((6, number_parameters))
        classical_drift = np.zeros((6, number_parameters))

        for parameter_id in range(number_parameters):
            joint_id = parameter_id // 6 + 1
            tangent = np.zeros(6)
            tangent[parameter_id % 6] = cls.finite_difference_step
            model_plus = model.copy()
            model_minus = model.copy()
            model_plus.jointPlacements[joint_id] = model.jointPlacements[
                joint_id
            ] * pin.exp6(tangent)
            model_minus.jointPlacements[joint_id] = model.jointPlacements[
                joint_id
            ] * pin.exp6(-tangent)

            plus = cls.evaluate(model_plus, q, v, frame_id, reference_frame)
            minus = cls.evaluate(model_minus, q, v, frame_id, reference_frame)
            placement[:, parameter_id] = pin.log6(
                minus[0].inverse() * plus[0]
            ).vector / (2.0 * cls.finite_difference_step)
            jacobian[:, :, parameter_id] = (plus[1] - minus[1]) / (
                2.0 * cls.finite_difference_step
            )
            spatial_drift[:, parameter_id] = (plus[2] - minus[2]) / (
                2.0 * cls.finite_difference_step
            )
            classical_drift[:, parameter_id] = (plus[3] - minus[3]) / (
                2.0 * cls.finite_difference_step
            )

        return placement, jacobian, spatial_drift, classical_drift

    @classmethod
    def finite_difference_directions(
        cls, model, q, v, frame_id, reference_frame, directions
    ):
        nominal = cls.evaluate(model, q, v, frame_id, reference_frame)
        derivatives = [np.zeros((6, directions.shape[1]))]
        derivatives.extend(
            np.zeros((*value.shape, directions.shape[1])) for value in nominal[1:]
        )
        for parameter_id in range(directions.shape[1]):
            model_plus = model.copy()
            model_minus = model.copy()
            for joint_id in range(1, model.njoints):
                tangent = directions[6 * (joint_id - 1) : 6 * joint_id, parameter_id]
                model_plus.jointPlacements[joint_id] = model.jointPlacements[
                    joint_id
                ] * pin.exp6(cls.finite_difference_step * tangent)
                model_minus.jointPlacements[joint_id] = model.jointPlacements[
                    joint_id
                ] * pin.exp6(-cls.finite_difference_step * tangent)
            plus = cls.evaluate(model_plus, q, v, frame_id, reference_frame)
            minus = cls.evaluate(model_minus, q, v, frame_id, reference_frame)
            derivatives[0][:, parameter_id] = pin.log6(
                minus[0].inverse() * plus[0]
            ).vector / (2.0 * cls.finite_difference_step)
            for derivative, plus_value, minus_value in zip(
                derivatives[1:], plus[1:], minus[1:]
            ):
                derivative[..., parameter_id] = (plus_value - minus_value) / (
                    2.0 * cls.finite_difference_step
                )
        return tuple(derivatives)

    def assert_direction_matches(self, model, seed):
        q, v = self.make_state(model, seed)
        frame_id = model.nframes - 1
        directions = np.random.default_rng(seed + 1).standard_normal(
            (6 * (model.njoints - 1), 3)
        )
        for reference_frame in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED, pin.WORLD):
            analytical = compute_frame_placement_derivatives(
                model,
                model.createData(),
                q,
                v,
                frame_id,
                directions,
                reference_frame,
            )
            numerical = self.finite_difference_directions(
                model, q, v, frame_id, reference_frame, directions
            )
            for analytical_value, numerical_value in zip(analytical, numerical):
                self.assertApprox(analytical_value, numerical_value, 2e-5)

    def test_every_placement_direction_and_reference_frame(self):
        model = pin.buildSampleModelManipulator()
        q, v = self.make_state(model, seed=1)
        frame_id = model.nframes - 1
        number_parameters = 6 * (model.njoints - 1)

        for reference_frame in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED, pin.WORLD):
            analytical = compute_frame_placement_derivatives_full(
                model,
                model.createData(),
                q,
                v,
                frame_id,
                reference_frame,
            )
            numerical = self.finite_difference(model, q, v, frame_id, reference_frame)
            self.assertEqual(analytical[0].shape, (6, number_parameters))
            self.assertEqual(analytical[1].shape, (6, model.nv, number_parameters))
            self.assertEqual(analytical[2].shape, (6, number_parameters))
            self.assertEqual(analytical[3].shape, (6, number_parameters))
            for analytical_value, numerical_value in zip(analytical, numerical):
                self.assertApprox(analytical_value, numerical_value, 2e-5)

    def test_floating_base_model(self):
        self.assert_direction_matches(pin.buildSampleModelHumanoidRandom(), seed=3)

    def test_stacked_parameter_directions(self):
        self.assert_direction_matches(pin.buildSampleModelManipulator(), seed=2)

    def test_mimic_joint_model(self):
        self.assert_direction_matches(pin.buildSampleModelManipulator(True), seed=5)

    def test_zero_velocity_has_zero_drift_derivatives(self):
        model = pin.buildSampleModelManipulator()
        q, _ = self.make_state(model, seed=7)
        _, _, spatial_drift, classical_drift = compute_frame_placement_derivatives_full(
            model,
            model.createData(),
            q,
            np.zeros(model.nv),
            model.nframes - 1,
        )
        self.assertApprox(spatial_drift, np.zeros_like(spatial_drift))
        self.assertApprox(classical_drift, np.zeros_like(classical_drift))

    def test_native_layout_and_reusable_workspace(self):
        model = pin.buildSampleModelManipulator()
        q, v = self.make_state(model, seed=8)
        frame_id = model.nframes - 1
        directions = np.random.default_rng(9).standard_normal(
            (6 * (model.njoints - 1), 3)
        )
        native_directions = np.zeros((6 * model.njoints, directions.shape[1]))
        native_directions[6:, :] = directions
        workspace = pin.FramePlacementDerivativesWorkspace(model.nv, 5)

        expected = compute_frame_placement_derivatives(
            model,
            model.createData(),
            q,
            v,
            frame_id,
            directions,
            pin.WORLD,
        )
        actual = compute_frame_placement_derivatives(
            model,
            model.createData(),
            q,
            v,
            frame_id,
            native_directions,
            pin.WORLD,
            workspace,
        )
        self.assertEqual(workspace.nv, model.nv)
        self.assertEqual(workspace.parameter_capacity, 5)
        for expected_value, actual_value in zip(expected, actual):
            self.assertApprox(expected_value, actual_value)

        with self.assertRaisesRegex(ValueError, "workspace"):
            compute_frame_placement_derivatives(
                model,
                model.createData(),
                q,
                v,
                frame_id,
                directions,
                workspace=pin.FramePlacementDerivativesWorkspace(model.nv, 2),
            )

    def test_input_validation(self):
        model = pin.buildSampleModelManipulator()
        data = model.createData()
        q, v = self.make_state(model, seed=9)
        with self.assertRaisesRegex(ValueError, "frame_id"):
            compute_frame_placement_derivatives_full(model, data, q, v, model.nframes)
        with self.assertRaisesRegex(ValueError, "q must have shape"):
            compute_frame_placement_derivatives_full(
                model, data, q[:-1], v, model.nframes - 1
            )
        with self.assertRaisesRegex(ValueError, "v must have shape"):
            compute_frame_placement_derivatives_full(
                model, data, q, v[:-1], model.nframes - 1
            )
        with self.assertRaisesRegex(ValueError, "reference_frame"):
            compute_frame_placement_derivatives_full(
                model, data, q, v, model.nframes - 1, 42
            )
        with self.assertRaisesRegex(ValueError, "joint_placement_jacobians"):
            compute_frame_placement_derivatives(
                model,
                data,
                q,
                v,
                model.nframes - 1,
                np.zeros((6 * (model.njoints - 1) - 1, 2)),
            )


if __name__ == "__main__":
    unittest.main()
