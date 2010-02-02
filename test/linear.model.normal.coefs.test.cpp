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

#include <cppmc/mcmc.normal.hpp>
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
// global rng generators
base_generator_type MCMCObject::generator_;
base_generator_type MCMCJumperBase::generator_;

int main() {
  const int N = 1000;
  mat X = rand<mat>(N,2);
  mat y = rand<mat>(N,1);

  // make X col 0 const
  for(int i = 0; i < N; i++) { X(i,0) = 1; }

  vec coefs;
  solve(coefs, X, y);
  Normal<vec> B(0.0, 10, vec(2));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<mat> likelihood(y, obs_fcst, 100);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e2, 4);
  const vector<vec>& coefs_hist(B.getHistory());

  vec avg_coefs(2);
  avg_coefs.fill(0);
  ofstream outfile;
  outfile.open ("coefs.csv");
  for(uint i = 0; i < coefs_hist.size(); i++) {
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
