#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>

#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/uniform.hpp>

#include <cppmc/mcmc.deterministic.hpp>
#include <cppmc/mcmc.uniform.hpp>
#include <cppmc/mcmc.normal.likelihood.hpp>

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
base_generator_type MCMCJumperBase::generator_;
base_generator_type MCMCObject::generator_;

int main() {
  const int N = 10;
  mat X = rand<mat>(N,2);
  mat y = rand<mat>(N,1);

  // make X col 0 const
  for(int i = 0; i < N; i++) { X(i,0) = 1; }

  mat coefs;
  solve(coefs, X, y);
  Uniform<double> B(-1.0,1.0, mat(2,1));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<double> likelihood(y, obs_fcst, 0.01);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e4, 4);
  const vector<mat>& coefs_hist(B.getHistory());

  mat avg_coefs(2,1);
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
