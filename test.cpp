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


inline int nrow(vec v) { return v.n_rows; }
inline int ncol(vec v) { return v.n_cols; }
inline int nrow(mat m) { return m.n_rows; }
inline int ncol(mat m) { return m.n_cols; }
//inline int size(vec v) { return nrow(v) * ncol(v); }
inline int size(vec m) { return nrow(m) * ncol(m); }

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
  mat eval() {
    return X_ * b_.exposeValue();
  }
};

// global rng generators
gsl_rng* MCMCJumperBase::rng_source_ =  NULL;
base_generator_type MCMCObject::generator_;

int main() {
  const gsl_rng_type* T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  MCMCJumperBase::setupRNG(gsl_rng_alloc(T));

  const int N = 1000;
  mat X(N,2);
  mat y(N,1);

  base_generator_type sample_generator;
  normal_distribution<double> nd(0,2);
  boost::variate_generator<base_generator_type&, normal_distribution<double> > rng_(sample_generator, nd);

  for(int i = 0; i < N; i++) {
    y[i] = rng_();
    X(i,0) = 1;
    X(i,1) = rng_();
  }
  vec coefs;
  solve(coefs, X, y);
  Uniform<vec> B(-1.0,1.0, vec(2));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<mat> likelihood(y, obs_fcst, 100);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e4, 4);
  const vector<vec>& coefs_hist(B.getHistory());

  vec avg_coefs(2);
  avg_coefs.fill(0);
  ofstream outfile;
  outfile.open ("coefs.csv");
  for(int i = 0; i < coefs_hist.size(); i++) {
    outfile << coefs_hist[i][0] << "," << coefs_hist[i][1] << endl;
    avg_coefs += coefs_hist[i];
  }
  avg_coefs /= coefs_hist.size();

  cout << "iterations: " << iterations << endl;
  cout << "collected " << coefs_hist.size() << " samples." << endl;
  cout << "lm coefs" << endl << coefs << endl;
  cout << "avg_coefs" << endl << avg_coefs << endl;
  return 1;
}
