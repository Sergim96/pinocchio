#
# Copyright (c) 2026 Heriot-Watt University
#

import unittest

import numpy as np
import pinocchio as pin
import pinocchio.cppadcg as cgpin
import pinocchio.cppadcg_float32 as cgpin32
from pycppad import ADCG, ADCGFloat32, ADCGFunFloat32, CGFloat32, Independent


class TestCppADCodeGenFloat32Bindings(unittest.TestCase):
    def test_coexistence_and_jacobian(self):
        self.assertIs(cgpin.ScalarType, ADCG)
        self.assertIs(cgpin32.ScalarType, ADCGFloat32)
        self.assertIsNot(cgpin32.Model, cgpin.Model)

        pin_model = pin.buildSampleModelHumanoidRandom()
        model = cgpin.Model(pin_model)
        model32 = cgpin32.Model(pin_model)
        self.assertEqual(model.nq, model32.nq)
        self.assertEqual(model.nv, model32.nv)

        data32 = model32.createData()
        nx = model32.nq + 2 * model32.nv
        x = np.array([ADCGFloat32(CGFloat32(0.0))] * nx)
        x[: model32.nq] = cgpin32.neutral(model32)
        Independent(x)

        y = cgpin32.rnea(
            model32,
            data32,
            x[: model32.nq],
            x[model32.nq : model32.nq + model32.nv],
            x[model32.nq + model32.nv :],
        )
        fun = ADCGFunFloat32(x, y)
        ind_vars = np.array([CGFloat32(1.0)] * nx)
        jacobian = fun.Jacobian(ind_vars)

        self.assertEqual(y.shape, (model32.nv,))
        self.assertEqual(jacobian.size, model32.nv * nx)


if __name__ == "__main__":
    unittest.main()
