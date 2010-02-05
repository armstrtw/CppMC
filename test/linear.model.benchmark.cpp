#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>

#include <armadillo>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/uniform.hpp>

#include <cppmc/mcmc.deterministic.hpp>
#include <cppmc/mcmc.uniform.hpp>
#include <cppmc/mcmc.normal.likelihood.hpp>

using namespace boost;
using namespace arma;
using namespace CppMC;
using std::vector;
using std::ofstream;
using std::cout;
using std::endl;
using boost::math::uniform;
typedef boost::minstd_rand base_generator_type;

class EstimatedY : public MCMCDeterministic<double> {
private:
  mat& X_;
  MCMCStochastic<double>& b_;
public:
  EstimatedY(Mat<double>& X, MCMCStochastic<double>& b): MCMCDeterministic<double>(X * b.exposeValue()), X_(X), b_(b) {
    registerParents();
  }
  void registerParents() {
    parents_.push_back(&b_);
  }
  mat eval() const {
    return X_ * b_.exposeValue();
  }
};

// global rng generators
base_generator_type MCMCObject::generator_;
base_generator_type MCMCJumperBase::generator_;

int main() {
  const int NR = 1000;
  const int NC = 2;
  mat X = rand<mat>(NR,NC);
  mat y = rand<mat>(NR,1);

  Uniform<double> B(-1.0,1.0, mat(NC,1));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<double> likelihood(y, obs_fcst, 100);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e2, 4);
  return 1;
}
