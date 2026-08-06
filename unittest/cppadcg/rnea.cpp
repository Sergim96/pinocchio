//
// Copyright (c) 2026 Heriot-Watt University
//

#include "pinocchio/codegen/cppadcg-algo.hpp"

#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/multibody/sample-models.hpp"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_rnea_code_generation)
{
  typedef double Scalar;
  typedef pinocchio::ModelTpl<Scalar> Model;
  typedef Model::Data Data;
  typedef Model::ConfigVectorType ConfigVectorType;
  typedef Model::TangentVectorType TangentVectorType;

  pinocchio::Model model_double;
  pinocchio::buildModels::humanoidRandom(model_double);
  Model model = model_double.cast<Scalar>();
  model.name = "humanoid_codegen";
  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill(1.);
  Data data(model);

  pinocchio::CodeGenRNEA<Scalar> codegen(model, "rnea", "cg_rnea_eval");
  codegen.initLib();
  codegen.compileAndLoadLib(PINOCCHIO_CXX_COMPILER);

  ConfigVectorType q = pinocchio::randomConfiguration(model);
  TangentVectorType v = TangentVectorType::Random(model.nv);
  TangentVectorType a = TangentVectorType::Random(model.nv);

  codegen.evalFunction(q, v, a);
  codegen.evalJacobian(q, v, a);

  pinocchio::crba(model, data, q, pinocchio::Convention::WORLD);
  data.M.triangularView<Eigen::StrictlyLower>() =
    data.M.transpose().triangularView<Eigen::StrictlyLower>();

  BOOST_CHECK(codegen.dtau_da.isApprox(data.M, 1e-8));
}

BOOST_AUTO_TEST_SUITE_END()
