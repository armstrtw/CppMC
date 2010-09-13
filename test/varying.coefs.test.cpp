#include <iostream>
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
  MCMCStochastic<double,Mat>& B_;
  mutable mat B_full_rank_;
  mat permutation_matrix_;
  mat row_sum_permutation_;
public:
  EstimatedY(Mat<double>& X, MCMCStochastic<double,Mat>& B, ivec& groups):
    MCMCDeterministic<double,Mat>(mat(X.n_rows,1)), X_(X), B_(B), B_full_rank_(X_.n_rows,X.n_cols),
    permutation_matrix_(X_.n_rows,B.nrow()) {
    permutation_matrix_.fill(0.0);

    for(uint i = 0; i < groups.n_elem; i++) {
      permutation_matrix_(i,groups[i]) = 1.0;
    }
    MCMCDeterministic<double,Mat>::value_ = eval();
  }
  void getParents(std::vector<MCMCObject*>& parents) const {
    parents.push_back(&B_);
  }
  Mat<double> eval() const {
    const mat& B = B_();
    B_full_rank_ = permutation_matrix_ * B;
    return sum(X_ % B_full_rank_,1);
  }
};

// global rng generators
CppMCGeneratorT MCMCObject::generator_;

int main() {
  const uint NR = 1000;
  const uint NC = 4;
  const uint J = 3;

  mat X = randn<mat>(NR,NC);
  mat y = randn<mat>(NR,1);

  // make X col 0 const
  for(uint i = 0; i < NR; i++) { X(i,0) = 1; }

  // create fake groups
  ivec groups(NR);
  for(uint i = 0; i < NR; i++) {
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

  Normal<Mat> B(0.0, 0.0001, randn<mat>(J,NC));
  
  EstimatedY obs_fcst(X, B, groups);
  Uniform<Mat> tauY(0, 100, vec(1)); tauY[0] = 1.0;
  NormalLikelihood<Mat> likelihood(y, obs_fcst, tauY);
  likelihood.print();
  likelihood.sample(1e5, 1e4, 10);

  cout << "collected " << B.getHistory().size() << " samples." << endl;
  cout << "avg_coefs" << endl << B.mean() << endl;

  return 1;
}
