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

class TauY : public MCMCDeterministic<double,Mat> {
private:
  Bernoulli<Col>& state_;
  Normal<Col>& sd0_;
  Normal<Col>& sd1_;
public:
  TauY(Bernoulli<Col>& state, Normal<Col>& sd0, Normal<Col>& sd1):
    MCMCDeterministic<double,Mat>(mat(state.size(),1)), state_(state), sd0_(sd0), sd1_(sd1)
  {
    for(uint i = 0; i < state_.size(); i++) {
      MCMCDeterministic<double,Mat>::value_[i] = state_[i] ? pow(sd1_[0],-2.0) : pow(sd0_[0],-2.0);
    }
  }
  void getParents(std::vector<MCMCObject*>& parents) const {
    parents.push_back(&state_);
    parents.push_back(&sd0_);
    parents.push_back(&sd1_);
  }
  Mat<double> eval() const {
    mat ans(state_.size(),1);
    for(uint i = 0; i < state_.size(); i++) {
      ans[i] = state_[i] ? pow(sd1_[0],-2) : pow(sd0_[0],-2);
    }
    return ans;
  }
};

class EstimatedY : public MCMCDeterministic<double,Mat> {
private:
  mat& X_;
  MCMCStochastic<double,Mat>& B_;
  mutable mat B_full_rank_;
  mutable mat permutation_matrix_;
  Bernoulli<Col>& state_;
public:
  EstimatedY(Mat<double>& X, MCMCStochastic<double,Mat>& B, Bernoulli<Col>& state):
    MCMCDeterministic<double,Mat>(mat(X.n_rows,1)), X_(X), B_(B), B_full_rank_(X_.n_rows,X.n_cols),
    permutation_matrix_(X_.n_rows,B.nrow()), state_(state) {
    permutation_matrix_.fill(0.0);
    MCMCDeterministic<double,Mat>::value_ = eval();
  }
  void update_perm_mat() const {
    for(uint i = 0; i < state_.nrow(); i++) {
      permutation_matrix_(i,state_[i]) = 1.0;
    }
  }
  void getParents(std::vector<MCMCObject*>& parents) const {
    parents.push_back(&B_);
    parents.push_back(&state_);
  }
  Mat<double> eval() const {
    const mat& B = B_();
    update_perm_mat();
    B_full_rank_ = permutation_matrix_ * B;
    return sum(X_ % B_full_rank_,1);
  }
};

// global rng generators
CppMCGeneratorT MCMCObject::generator_;

int main() {
  const int NR = 20;
  const int NC = 2;
  const int Nstates = 2;
  double true_sd0 = 1.0;
  double true_sd1 = 3.0;
  double state1p = 0.05;

  // create state probability vector, and establish true state variable based on it
  vec statetest = randu<vec>(NR);
  ivec true_state(NR);
  for(int i = 0; i < NR; i++) { true_state[i] = statetest[i] > (1-state1p) ? 1 : 0; }

  mat X = randn<mat>(NR,NC);
  mat y = randn<mat>(NR,1);

  // scale y based on true state vector
  for(int i = 0; i < NR; i++) { y[i] *= (true_state[i] > 0) ? true_sd1 : true_sd0; }
  cout << "y sd:" << stddev(y,0) << endl;

  // make X col 0 const
  for(int i = 0; i < NR; i++) { X(i,0) = 1; }

  Normal<Mat> B(0.0, 0.0001, randn<mat>(Nstates,NC)); B.print();
  Normal<Col> sd0(1.0,0.0001,randu<vec>(1)); sd0.print();
  Normal<Col> sd1(2.0,0.0001,randu<vec>(1)); sd1.print();
  Uniform<Col> high_vol_statep(0,0.10, randu<vec>(1)); high_vol_statep.print();
  Bernoulli<Col> state(high_vol_statep,(randu<vec>(NR) > .5)); state.print();
  EstimatedY obs_fcst(X, B, state); obs_fcst.print();
  TauY tau_y(state,sd0,sd1); tau_y.print();
  NormalLikelihood<Mat> likelihood(y, obs_fcst, tau_y);

  int iterations = 1e6;
  likelihood.sample(iterations, 1e5, 100);

  cout << "iterations: " << iterations << endl;
  cout << "collected " << B.getHistory().size() << " samples." << endl;
  //cout << "lm coefs" << endl << coefs << endl;
  cout << "avg_coefs" << endl << B.mean() << endl;
  cout << "state" << endl << state.mean() << endl;
  // cout << "tau: " << tau_y.mean() << endl;
  return 1;
}
