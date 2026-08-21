import unittest

import numpy as np
import pinocchio as pin
from pinocchio.derivative.contact_kinematic import (
    compute_rigid_constraint_kinematic_derivatives,
)
from test_case import PinocchioTestCase as TestCase


class TestContactKinematicDerivatives(TestCase):
    finite_difference_step = 1e-7

    @staticmethod
    def make_state(model, seed):
        random = np.random.default_rng(seed)
        q = pin.integrate(
            model, pin.neutral(model), 0.25 * random.standard_normal(model.nv)
        )
        return q, random.standard_normal(model.nv)

    @staticmethod
    def evaluate(model, q, v, constraint_model):
        data = model.createData()
        pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))
        pin.computeJointJacobians(model, data)
        constraint_data = constraint_model.createData()
        constraint_model.calc(model, data, constraint_data)

        joint1_id = constraint_model.joint1_id
        joint2_id = constraint_model.joint2_id
        placement1 = constraint_model.joint1_placement
        placement2 = constraint_model.joint2_placement
        jacobian1 = pin.getFrameJacobian(model, data, joint1_id, placement1, pin.LOCAL)
        jacobian2 = pin.getFrameJacobian(model, data, joint2_id, placement2, pin.LOCAL)
        c1Mc2 = constraint_data.c1Mc2

        if constraint_model.type == pin.ContactType.CONTACT_6D:
            local_jacobian = jacobian1 - c1Mc2.action @ jacobian2
            placement_error = -pin.log6(c1Mc2).vector
        else:
            local_jacobian = jacobian1[:3] - c1Mc2.rotation @ jacobian2[:3]
            placement_error = -c1Mc2.translation

        if constraint_model.reference_frame == pin.LOCAL_WORLD_ALIGNED:
            rotation = constraint_data.oMc1.rotation
            if constraint_model.type == pin.ContactType.CONTACT_6D:
                output_rotation = np.zeros((6, 6))
                output_rotation[:3, :3] = rotation
                output_rotation[3:, 3:] = rotation
                jacobian = output_rotation @ local_jacobian
                placement_error = output_rotation @ placement_error
            else:
                jacobian = rotation @ local_jacobian
                placement_error = rotation @ placement_error
        else:
            jacobian = local_jacobian

        acceleration1 = pin.getFrameAcceleration(
            model, data, joint1_id, placement1, pin.LOCAL
        )
        acceleration2 = pin.getFrameAcceleration(
            model, data, joint2_id, placement2, pin.LOCAL
        )
        if constraint_model.type == pin.ContactType.CONTACT_6D:
            if constraint_model.reference_frame == pin.LOCAL:
                velocity1 = pin.Motion(jacobian1 @ v)
                velocity2 = pin.Motion(jacobian2 @ v)
                velocity2_in_1 = c1Mc2.act(velocity2)
                velocity_error = velocity1 - velocity2_in_1
                drift = (
                    acceleration1
                    + velocity_error.cross(velocity2_in_1)
                    - c1Mc2.act(acceleration2)
                ).vector
            else:
                local_drift = acceleration1 - c1Mc2.act(acceleration2)
                drift = output_rotation @ local_drift.vector
        else:
            classical1 = pin.getFrameClassicalAcceleration(
                model, data, joint1_id, placement1, pin.LOCAL
            )
            classical2 = pin.getFrameClassicalAcceleration(
                model, data, joint2_id, placement2, pin.LOCAL
            )
            drift = classical1.linear - c1Mc2.rotation @ classical2.linear
            if constraint_model.reference_frame == pin.LOCAL_WORLD_ALIGNED:
                drift = constraint_data.oMc1.rotation @ drift

        return (
            c1Mc2.copy(),
            placement_error.copy(),
            jacobian.copy(),
            (jacobian @ v).copy(),
            drift.copy(),
        )

    @classmethod
    def finite_difference(cls, model, q, v, constraint_model, directions):
        nominal = cls.evaluate(model, q, v, constraint_model)
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

            plus = cls.evaluate(model_plus, q, v, constraint_model)
            minus = cls.evaluate(model_minus, q, v, constraint_model)
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

    def check_constraint(self, model, joint1_id, joint2_id, seed, stabilized=False):
        q, v = self.make_state(model, seed)
        random = np.random.default_rng(seed + 1)
        directions = 0.2 * random.standard_normal((6 * (model.njoints - 1), 3))
        placement1 = pin.SE3.Random()
        placement2 = pin.SE3.Random()

        for contact_type in (
            pin.ContactType.CONTACT_3D,
            pin.ContactType.CONTACT_6D,
        ):
            for reference_frame in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED):
                constraint_model = pin.RigidConstraintModel(
                    contact_type,
                    model,
                    joint1_id,
                    placement1,
                    joint2_id,
                    placement2,
                    reference_frame,
                )
                if stabilized:
                    constraint_model.setBaumgarteCorrectorParameters(
                        pin.BaumgarteCorrectorParameters(3.0, 0.7)
                    )
                constraint_data = constraint_model.createData()
                analytical = compute_rigid_constraint_kinematic_derivatives(
                    model,
                    model.createData(),
                    constraint_model,
                    constraint_data,
                    q,
                    v,
                    directions,
                )
                numerical = self.finite_difference(
                    model, q, v, constraint_model, directions
                )
                constraint_size = constraint_model.residualSize()
                self.assertEqual(analytical[0].shape, (6, 3))
                self.assertEqual(analytical[1].shape, (constraint_size, 3))
                self.assertEqual(analytical[2].shape, (constraint_size, model.nv, 3))
                self.assertEqual(analytical[3].shape, (constraint_size, 3))
                self.assertEqual(analytical[4].shape, (constraint_size, 3))
                for analytical_value, numerical_value in zip(analytical, numerical):
                    self.assertLess(
                        np.max(np.abs(analytical_value - numerical_value)), 3e-5
                    )
                if stabilized:
                    analytical_corrector = -(3.0 * analytical[1] + 0.7 * analytical[3])
                    numerical_corrector = -(3.0 * numerical[1] + 0.7 * numerical[3])
                    self.assertLess(
                        np.max(np.abs(analytical_corrector - numerical_corrector)),
                        3e-5,
                    )

    def test_two_moving_branches(self):
        model = pin.buildSampleModelHumanoidRandom()
        self.check_constraint(
            model,
            model.getJointId("rarm2_joint"),
            model.getJointId("larm2_joint"),
            seed=1,
        )

    def test_moving_frame_against_universe(self):
        model = pin.buildSampleModelManipulator()
        self.check_constraint(model, model.njoints - 1, 0, seed=3)

    def test_stabilized_mimic_constraint(self):
        model = pin.buildSampleModelManipulator(True)
        self.check_constraint(
            model,
            model.njoints - 1,
            max(1, model.njoints // 2),
            seed=5,
            stabilized=True,
        )

    def test_reusable_workspace_and_native_layout(self):
        model = pin.buildSampleModelManipulator()
        q, v = self.make_state(model, 8)
        constraint_model = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            model,
            model.njoints - 1,
            pin.SE3.Random(),
            1,
            pin.SE3.Random(),
            pin.LOCAL,
        )
        directions = np.random.default_rng(9).standard_normal(
            (6 * (model.njoints - 1), 2)
        )
        native_directions = np.zeros((6 * model.njoints, 2))
        native_directions[6:] = directions
        workspace = pin.RigidConstraintKinematicDerivativesWorkspace(model.nv, 4)
        expected = compute_rigid_constraint_kinematic_derivatives(
            model,
            model.createData(),
            constraint_model,
            constraint_model.createData(),
            q,
            v,
            directions,
        )
        actual = compute_rigid_constraint_kinematic_derivatives(
            model,
            model.createData(),
            constraint_model,
            constraint_model.createData(),
            q,
            v,
            native_directions,
            workspace,
        )
        for expected_value, actual_value in zip(expected, actual):
            self.assertApprox(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()
