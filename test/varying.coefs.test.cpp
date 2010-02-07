#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>

#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/uniform.hpp>

#include <cppmc/mcmc.hyperprior.hpp>
#include <cppmc/mcmc.deterministic.hpp>
#include <cppmc/mcmc.uniform.hpp>
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
  const int NR = 1000;
  const int NC = 2;
  const int J = 4;
  
  mat X = rand<mat>(NR,NC);
  mat y = rand<mat>(NR,1);

  // make X col 0 const
  for(int i = 0; i < NR; i++) { X(i,0) = 1; }

  vec coefs;
  solve(coefs, X, y);

  HyperPrior<double,Col> mu_mean(0.0);
  HyperPrior<double,Col> mu_sd(1.0);
  HyperPrior<double,Col> sd_mean(1.0);
  HyperPrior<double,Col> sd_sd(1.0);

  Normal<double,Col,Col> mu_b(mu_mean, mu_sd, vec(NC));
  Normal<double,Col,Col> sd_b(sd_mean, sd_sd, vec(NC));
  Normal<double,Col,Mat> B(mu_b, sd_b, mat(J,NC));

  EstiMatedY obs_fcst(X, B);
  NormalLikelihood<Mat> likelihood(y, obs_fcst, 100);

  int iterations = 1e5;
  likelihood.sample(iterations, 1e4, 4);
  const vector<vec>& coefs_hist(B.getHistory());

  vec avg_coefs(NC);
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
