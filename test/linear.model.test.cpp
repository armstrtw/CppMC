#include <iostream>
#include <fstream>
#include <vector>
#include <boost/random.hpp>
#include <cppmc/cppmc.hpp>

using namespace CppMC;
using std::vector;
using std::ofstream;
using std::cout;
using std::endl;

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
CppMCGeneratorT MCMCObject::generator_;

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
  //likelihood.print();
  likelihood.sample(iterations, 1e4, 10);

  cout << "iterations: " << iterations << endl;
  cout << "collected " << B.getHistory().size() << " samples." << endl;
  cout << "lm coefs" << endl << coefs << endl;
  cout << "avg_coefs" << endl << B.mean() << endl;
  return 1;
}
