#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>

#include <armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/uniform.hpp>

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


inline uint nrow(vec v) { return v.n_rows; }
inline uint ncol(vec v) { return v.n_cols; }
inline uint nrow(mat m) { return m.n_rows; }
inline uint ncol(mat m) { return m.n_cols; }
//inline uint size(vec v) { return nrow(v) * ncol(v); }
inline uint size(vec m) { return nrow(m) * ncol(m); }

class EstimatedY : public MCMCDeterministic<mat> {
private:
  mat& X_;
  MCMCStochastic<vec>& b_;
public:
  EstimatedY(mat& X, MCMCStochastic<vec>& b): MCMCDeterministic<mat>(X * b.exposeValue()), X_(X), b_(b) {
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
  const int NC = 20;
  mat X = rand<mat>(NR,NC);
  mat y = rand<mat>(NR,1);

  Uniform<vec> B(-1.0,1.0, vec(NC));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<mat> likelihood(y, obs_fcst, 100);
  int iterations = 1e4;
  likelihood.sample(iterations, 1e2, 4);
  return 1;
}
