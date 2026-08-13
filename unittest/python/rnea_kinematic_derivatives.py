import unittest

import numpy as np
import pinocchio as pin
from pinocchio.derivative.rnea_kinematic import (
    compute_rnea_placement_derivatives,
)
from test_case import PinocchioTestCase as TestCase


class TestRNEAKinematicDerivatives(TestCase):
    finite_difference_step = 1e-7

    @staticmethod
    def make_state(model, seed):
        random = np.random.default_rng(seed)
        q = pin.integrate(
            model, pin.neutral(model), 0.3 * random.standard_normal(model.nv)
        )
        v = random.standard_normal(model.nv)
        a = random.standard_normal(model.nv)
        return q, v, a

    @classmethod
    def finite_difference(cls, model, q, v, a, placement_jacobians):
        number_parameters = placement_jacobians.shape[2]
        derivative = np.zeros((model.nv, number_parameters))

        for parameter_id in range(number_parameters):
            model_plus = model.copy()
            model_minus = model.copy()

            for joint_id in range(1, model.njoints):
                tangent = placement_jacobians[joint_id, :, parameter_id]
                model_plus.jointPlacements[joint_id] = model.jointPlacements[
                    joint_id
                ] * pin.exp6(cls.finite_difference_step * tangent)
                model_minus.jointPlacements[joint_id] = model.jointPlacements[
                    joint_id
                ] * pin.exp6(-cls.finite_difference_step * tangent)

            torque_plus = pin.rnea(model_plus, model_plus.createData(), q, v, a).copy()
            torque_minus = pin.rnea(
                model_minus, model_minus.createData(), q, v, a
            ).copy()
            derivative[:, parameter_id] = (torque_plus - torque_minus) / (
                2.0 * cls.finite_difference_step
            )

        return derivative

    def assert_matches_finite_difference(self, model, q, v, a, placement_jacobians):
        analytic = compute_rnea_placement_derivatives(
            model,
            model.createData(),
            q,
            v,
            a,
            placement_jacobians,
        )
        numerical = self.finite_difference(model, q, v, a, placement_jacobians)
        self.assertApprox(analytic, numerical, 2e-5)

    def test_translation_and_rotation(self):
        model = pin.buildSampleModelManipulator()
        q, v, a = self.make_state(model, seed=1)
        placement_jacobians = np.zeros((model.njoints, 6, 2))

        placement_jacobians[2, 0, 0] = 1.0
        placement_jacobians[4, 5, 1] = 1.0

        self.assert_matches_finite_difference(model, q, v, a, placement_jacobians)

    def test_parameters_shared_by_several_placements(self):
        model = pin.buildSampleModelManipulator()
        q, v, a = self.make_state(model, seed=2)
        random = np.random.default_rng(3)
        placement_jacobians = 0.2 * random.standard_normal((model.njoints, 6, 4))
        placement_jacobians[0] = 0.0

        self.assert_matches_finite_difference(model, q, v, a, placement_jacobians)

    def test_floating_base_model(self):
        model = pin.buildSampleModelHumanoidRandom()
        q, v, a = self.make_state(model, seed=4)
        random = np.random.default_rng(5)
        placement_jacobians = 0.1 * random.standard_normal((model.njoints, 6, 3))
        placement_jacobians[0] = 0.0

        self.assert_matches_finite_difference(model, q, v, a, placement_jacobians)

    def test_mimic_joint_model(self):
        model = pin.buildSampleModelManipulator(True)
        q, v, a = self.make_state(model, seed=6)
        random = np.random.default_rng(7)
        placement_jacobians = 0.1 * random.standard_normal((model.njoints, 6, 3))
        placement_jacobians[0] = 0.0

        self.assert_matches_finite_difference(model, q, v, a, placement_jacobians)

    def test_reusable_native_workspace(self):
        model = pin.buildSampleModelManipulator()
        q, v, a = self.make_state(model, seed=11)
        random = np.random.default_rng(12)
        number_parameters = 3
        placement_jacobians = 0.1 * random.standard_normal(
            (model.njoints, 6, number_parameters)
        )
        placement_jacobians[0] = 0.0

        workspace = pin.RNEAPlacementDerivativesWorkspace(
            model.njoints, number_parameters + 2
        )
        result = compute_rnea_placement_derivatives(
            model,
            model.createData(),
            q,
            v,
            a,
            placement_jacobians,
            workspace,
        )
        result_without_workspace = compute_rnea_placement_derivatives(
            model, model.createData(), q, v, a, placement_jacobians
        )

        self.assertEqual(workspace.njoints, model.njoints)
        self.assertEqual(workspace.parameter_capacity, number_parameters + 2)
        self.assertApprox(result, result_without_workspace)

    def test_workspace_dtype_includes_model_scalar(self):
        model = pin.buildSampleModelManipulator()
        random = np.random.default_rng(8)
        q = np.zeros(model.nq, dtype=np.int64)
        v = np.arange(1, model.nv + 1, dtype=np.int64)
        a = -np.arange(1, model.nv + 1, dtype=np.int64)
        placement_jacobians = random.integers(
            -1, 2, size=(model.njoints, 6, 2), dtype=np.int64
        )
        placement_jacobians[0] = 0

        data = model.createData()
        derivative_from_integers = compute_rnea_placement_derivatives(
            model, data, q, v, a, placement_jacobians
        )
        derivative_from_float64 = compute_rnea_placement_derivatives(
            model,
            model.createData(),
            q.astype(np.float64),
            v.astype(np.float64),
            a.astype(np.float64),
            placement_jacobians.astype(np.float64),
        )
        self.assertEqual(derivative_from_integers.dtype, data.tau.dtype)
        self.assertApprox(derivative_from_integers, derivative_from_float64)

        q_float32, v_float32, a_float32 = self.make_state(model, seed=9)
        placement_jacobians_float32 = 0.1 * random.standard_normal(
            (model.njoints, 6, 2), dtype=np.float32
        )
        data = model.createData()
        derivative_from_float32 = compute_rnea_placement_derivatives(
            model,
            data,
            q_float32.astype(np.float32),
            v_float32.astype(np.float32),
            a_float32.astype(np.float32),
            placement_jacobians_float32,
        )
        derivative_from_promoted_float32 = compute_rnea_placement_derivatives(
            model,
            model.createData(),
            q_float32.astype(np.float64),
            v_float32.astype(np.float64),
            a_float32.astype(np.float64),
            placement_jacobians_float32.astype(np.float64),
        )
        self.assertEqual(derivative_from_float32.dtype, data.tau.dtype)
        self.assertApprox(
            derivative_from_float32, derivative_from_promoted_float32, 2e-5
        )

    def test_invalid_placement_jacobian_shape(self):
        model = pin.buildSampleModelManipulator()
        q, v, a = self.make_state(model, seed=10)

        with self.assertRaisesRegex(ValueError, "must have shape"):
            compute_rnea_placement_derivatives(
                model,
                model.createData(),
                q,
                v,
                a,
                np.zeros((model.njoints, 6)),
            )


if __name__ == "__main__":
    unittest.main()
