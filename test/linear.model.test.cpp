#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>

#include <boost/random.hpp>
#include <cppmc/cppmc.hpp>

using namespace CppMC;
using std::vector;
using std::ofstream;
using std::cout;
using std::endl;
typedef boost::minstd_rand base_generator_type;

class EstimatedY : public MCMCDeterministic<double,Mat> {
private:
  mat& X_;
  MCMCStochastic<double,Col>& b_;
public:
  EstimatedY(Mat<double>& X, MCMCStochastic<double,Col>& b): MCMCDeterministic<double,Mat>(X * b.exposeValue()), X_(X), b_(b) {
    registerParents();
  }
  void registerParents() {
    parents_.push_back(&b_);
  }
  Mat<double> eval() const {
    return X_ * b_.exposeValue();
  }
};

// global rng generators
base_generator_type MCMCJumperBase::generator_;
base_generator_type MCMCObject::generator_;

int main() {
  const int N = 1000;
  mat X = rand<mat>(N,2);
  mat y = rand<mat>(N,1);

  // make X col 0 const
  for(int i = 0; i < N; i++) { X(i,0) = 1; }

  vec coefs;
  solve(coefs, X, y);
  Uniform<Col> B(-1.0, 1.0, vec(2));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<Mat> likelihood(y, obs_fcst, 1);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e4, 1);
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
