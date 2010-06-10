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
  EstimatedY(Mat<double>& X, MCMCStochastic<double,Col>& b): MCMCDeterministic<double,Mat>(X * b()), X_(X), b_(b)
  {
    //cout << "b_.shape()" << b_.nrow() << " " << b_.ncol() << endl;
    //cout << "X_.shape()" << X_.n_rows << " " << X_.n_cols << endl;
  }
  void getParents(std::vector<MCMCObject*>& parents) const {
    parents.push_back(&b_);
  }
  Mat<double> eval() const {
    //cout << b_() << endl;
    return X_ * b_();
  }
};

// global rng generators
CppMCGeneratorT MCMCObject::generator_;

int main() {
  const int N = 100;
  mat X = rand<mat>(N,2);
  mat y = rand<mat>(N,1);

  // make X col 0 const
  for(int i = 0; i < N; i++) { X(i,0) = 1; }

  vec coefs;
  solve(coefs, X, y);
  Uniform<Col> B(-100.0, 100.0, rand<vec>(2));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<Mat> likelihood(y, obs_fcst, 1.0);

  cout << "B" << endl; B.print();
  cout << "obs_fcst" << endl << obs_fcst.eval() << endl;
  cout << "likelihood"; likelihood.print();

  int iterations = 1e5;
  likelihood.sample(iterations, 1e4, 10);

  cout << "iterations: " << iterations << endl;
  cout << "collected " << B.getHistory().size() << " samples." << endl;
  cout << "lm coefs" << endl << coefs << endl;
  cout << "avg_coefs" << endl << B.mean() << endl;
  return 1;
}
