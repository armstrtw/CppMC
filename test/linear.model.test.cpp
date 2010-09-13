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
  mat X = randn<mat>(N,2);
  mat y = randn<mat>(N,1);
  cout << "y sd:" << stddev(y,0) << endl;

  // make X col 0 const
  for(int i = 0; i < N; i++) { X(i,0) = 1; }

  vec coefs;
  solve(coefs, X, y);
  //Uniform<Col> B(-100.0, 100.0, rand<vec>(2));
  Normal<Col> B(0.0, 0.0001, randn<vec>(2));
  EstimatedY obs_fcst(X, B);
  Uniform<Mat> tauY(0, 100, vec(1)); tauY[0] = 1.0;
  NormalLikelihood<Mat> likelihood(y, obs_fcst, tauY);

  //cout << "B" << endl; B.print();
  //cout << "obs_fcst" << endl << obs_fcst.eval() << endl;
  //cout << "likelihood"; likelihood.print();

  int iterations = 1e5;
  likelihood.sample(iterations, 1e4, 100);

  cout << "iterations: " << iterations << endl;
  cout << "collected " << B.getHistory().size() << " samples." << endl;
  cout << "lm coefs" << endl << coefs << endl;
  cout << "avg_coefs" << endl << B.mean() << endl;
  cout << "tau: " << tauY.mean() << endl;
  return 1;
}
