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
  MCMCStochastic<double,Mat>& B_;
  mutable mat B_full_rank_;
  mat permutation_matrix_;
  mat row_sum_permutation_;
public:
  EstimatedY(Mat<double>& X, MCMCStochastic<double,Mat>& B, ivec& groups):
    MCMCDeterministic<double,Mat>(mat(X.n_rows,X.n_cols)), X_(X), B_(B), B_full_rank_(X_.n_rows,X.n_cols),
    permutation_matrix_(X_.n_rows,B.nrow()), row_sum_permutation_(B_full_rank_.n_cols,1) {
    registerParents();
    permutation_matrix_.fill(0.0);
    row_sum_permutation_.fill(1.0);
    for(uint i = 0; i < groups.n_elem; i++) {
      permutation_matrix_(i,groups[i]) = 1.0;
    }
    //cout << "permutation_matrix_" << endl << permutation_matrix_;
  }
  void registerParents() {
    parents_.push_back(&B_);
  }
  Mat<double> eval() const {
    const mat& B = B_.exposeValue();
    //cout << B << endl;;
    B_full_rank_ = permutation_matrix_ * B;

    //which is faster?    
    //return X_ % B_full_rank_ * row_sum_permutation_;
    return sum(X_ % B_full_rank_,1);
  }
};

// global rng generators
base_generator_type MCMCJumperBase::generator_;
base_generator_type MCMCObject::generator_;

int main() {
  const int NR = 1000;
  const int NC = 4;
  const int J = 3;

  mat X = rand<mat>(NR,NC);
  mat y = rand<mat>(NR,1);

  // make X col 0 const
  for(int i = 0; i < NR; i++) { X(i,0) = 1; }

  // create fake groups
  ivec groups(NR);
  for(int i = 0; i < NR; i++) {
    groups[i] = i % J;
  }

  // shift y's by group sums
  vec group_shift(J);
  for(uint i = 0; i < J; i++) {
    group_shift[i] = (i + 1) * 10;
  }
  cout << "group_shift" << endl << group_shift;

  // do the shift on the data
  for(uint i = 0; i < NR; i++) {
    y[i] += group_shift[ groups[i] ];
  }

  vec coefs;
  solve(coefs, X, y);

  HyperPrior<double,Col> mu_mean(0.0);
  HyperPrior<double,Col> mu_sd(1.0);
  HyperPrior<double,Col> sd_mean(1.0);
  HyperPrior<double,Col> sd_sd(1.0);

  Normal<Col> mu_b(mu_mean, mu_sd, vec(NC));
  Normal<Col> sd_b(sd_mean, sd_sd, vec(NC));
  Normal<Mat> B(mu_b, sd_b, mat(J,NC));
  
  EstimatedY obs_fcst(X, B, groups);
  NormalLikelihood<Mat> likelihood(y, obs_fcst, 1);
  likelihood.sample(1e5, 1e4, 4);

  const vector<mat>& coefs_hist(B.getHistory());
  mat avg_coefs(J,NC); avg_coefs.fill(0);
  for(uint i = 0; i < coefs_hist.size(); i++) {
    //outfile << coefs_hist[i][0] << "," << coefs_hist[i][1] << endl;
    avg_coefs += coefs_hist[i];
  }
  avg_coefs /= coefs_hist.size();
  cout << "collected " << coefs_hist.size() << " samples." << endl;
  cout << "avg_coefs" << endl << avg_coefs << endl;

  return 1;
}
