import numpy as np
import pinocchio as pin
import pinocchio.cppadcg_float32 as cgpin
from pycppad import (
    ADCGFloat32,
    ADCGFunFloat32,
    CGFloat32,
    CodeHandlerFloat32,
    Independent,
    LangCDefaultVariableNameGeneratorFloat32,
    LanguageCFloat32,
)

pinmodel = pin.buildSampleModelHumanoidRandom()
model = cgpin.Model(pinmodel)
data = model.createData()

nq = model.nq
nv = model.nv

x = np.array([ADCGFloat32(CGFloat32(0.0))] * (nq + nv + nv))
x[:nq] = cgpin.neutral(model)
Independent(x)

y = cgpin.rnea(model, data, x[:nq], x[nq : nq + nv], x[nq + nv :])

fun = ADCGFunFloat32(x, y)

# /***************************************************************************
# *                        Generate the C source code
# **************************************************************************/

# /**
# * start the special steps for source code generation for a Jacobian
# */
handler = CodeHandlerFloat32(50)

indVars = np.array([CGFloat32(1.0)] * (nq + nv + nv))
handler.makeVariables(indVars)

jac = fun.Jacobian(indVars)

langC = LanguageCFloat32("float", 3)
nameGen = LangCDefaultVariableNameGeneratorFloat32("y", "x", "v", "array", "sarray")
code = handler.generateCode(langC, jac, nameGen, "source")
print(code)
