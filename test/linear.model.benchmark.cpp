#include <iostream>
#include <fstream>
#include <vector>
#include <boost/random.hpp>
#include <cppmc/cppmc.hpp>

using namespace boost;
using namespace arma;
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
  EstimatedY(Mat<double>& X, MCMCStochastic<double,Col>& b): MCMCDeterministic<double,Mat>(X * b()), X_(X), b_(b)
  {}
  void getParents(std::vector<MCMCObject*>& parents) const {
    parents.push_back(&b_);
  }
  Mat<double> eval() const {
    return X_ * b_();
  }
};

// global rng generators
CppMCGeneratorT MCMCObject::generator_;

int main() {
  const int NR = 1000;
  const int NC = 2;
  mat X = randn<mat>(NR,NC);
  mat y = randn<mat>(NR,1);

  Uniform<Col> B(-100.0,100.0, randn<vec>(NC));
  EstimatedY obs_fcst(X, B);
  Uniform<Mat> tauY(0, 100, vec(1)); tauY[0] = 1.0;
  NormalLikelihood<Mat> likelihood(y, obs_fcst, tauY);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e2, 4);
  return 1;
}
