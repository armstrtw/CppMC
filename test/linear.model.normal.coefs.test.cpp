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
  const int N = 1000;
  const int NC = 2;
  mat X = rand<mat>(N,NC);
  mat y = rand<mat>(N,1);

  // make X col 0 const
  for(int i = 0; i < N; i++) { X(i,0) = 1; }

  vec coefs;
  solve(coefs, X, y);
  Normal<Col> B(0.0, 1.0, rand<vec>(NC));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<Mat> likelihood(y, obs_fcst, 1.0);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e2, 4);
  const vector<vec>& coefs_hist(B.getHistory());

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
