#include <iostream>
#include <fstream>
#include <vector>
#include <boost/random.hpp>

typedef boost::minstd_rand base_generator_type;
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
base_generator_type MCMCObject::generator_;

int main() {
  const int NR = 1000;
  const int NC = 2;
  mat X = rand<mat>(NR,NC);
  mat y = rand<mat>(NR,1);

  Uniform<Col> B(-1.0,1.0, vec(NC));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<Mat> likelihood(y, obs_fcst, 100);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e2, 4);
  return 1;
}
