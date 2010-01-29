#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>

#include <armadillo>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/uniform.hpp>

using namespace boost;
using namespace arma;
using std::vector;
using std::ofstream;
using std::cout;
using std::endl;
using boost::math::uniform;
typedef boost::minstd_rand base_generator_type;

const double neg_inf(-std::numeric_limits<double>::infinity());

inline int nrow(vec v) { return v.n_rows; }
inline int ncol(vec v) { return v.n_cols; }
inline int nrow(mat m) { return m.n_rows; }
inline int ncol(mat m) { return m.n_cols; }
//inline int size(vec v) { return nrow(v) * ncol(v); }
inline int size(vec m) { return nrow(m) * ncol(m); }

class MCMCJumperBase {
protected:
  static gsl_rng* rng_source_;
public:
  MCMCJumperBase() {}
  static void setupRNG(gsl_rng* rng) {
    rng_source_ = rng;
  }
};

template<typename T>
class MCMCJumper : public MCMCJumperBase {
 private:
  T old_value_;
  T sd_;
  T& value_;
  double scale_;
  void drawRNG() {
    for(int i = 0; i < nrow(value_) * ncol(value_); i++) {
      value_[i] += scale_ * gsl_ran_gaussian(rng_source_, sd_[i]);
    }
  }
 public:
  MCMCJumper(T& value): value_(value), old_value_(value), sd_(nrow(value), ncol(value)), scale_(1) {
    sd_.fill(1);
  }
  void setSD(const double sd) {
    for(int i = 0; i < ncol(sd_) * nrow(sd_); i++) {
      sd_[i] = sd;
    }
  }
  void setScale(const double scale) {
    scale_ = scale;
  }
  void jump() {
    old_value_ = value_;
    drawRNG();
  }
  void revert() {
    value_ = old_value_;
  }

  void tune(const double acceptance_rate) {
    //cout << "acceptance_rate: " << acceptance_rate << endl;
    if(acceptance_rate < .01) {
      scale_ *= .1;
      return;
    }
    if(acceptance_rate < .05) {
      scale_ *= .5;
      return;
    }
    if(acceptance_rate < .2) {
      scale_ *= .9;
      return;
    }
    if(acceptance_rate < .5) {
      scale_ *= 1.1;
      return;
    }
    if(acceptance_rate > .95) {
      scale_ *= 10;
      return;
    }
    if(acceptance_rate > .75) {
      scale_ *= 2;
      return;
    }
  }
};

class MCMCObject {
protected:
  static base_generator_type generator_;
public:
  MCMCObject() {}
  virtual double logp() const = 0;
  virtual void jump(int current_iteration) = 0;
  virtual void revert() = 0;
  virtual void tally() = 0;
  virtual void tally_parents() = 0;
  virtual void tune(const double acceptance_rate) = 0;
};

template<typename T>
class MCMCSpecialized : public MCMCObject {
protected:
  T value_;
  vector<T> history_;
public:
  MCMCSpecialized(const T& shape): MCMCObject(), value_(shape) {}
  const T& exposeValue() const {
    return value_;
  }
  void tally() {
    history_.push_back(value_);
    tally_parents();
  }
  const vector<T>& getHistory() const {
    return history_;
  }
};

template<typename T>
class MCMCDeterministic : public MCMCSpecialized<T> {
protected:
  vector<MCMCObject*> parents_;
public:
  MCMCDeterministic(const T& initial_value): MCMCSpecialized<T>(initial_value) {}
  double logp() const {
    double ans(0);
    for(vector<MCMCObject*>::const_iterator  iter = parents_.begin(); iter!=parents_.end(); iter++) {
      ans += (*iter)->logp();
    }
    return ans;
  }
  void jump(int current_iteration) {
    for(vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
      (*iter)->jump(current_iteration);
    }
    MCMCSpecialized<T>::value_ = eval();
  }
  void revert() {
    for(vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
      (*iter)->revert();
    }
    MCMCSpecialized<T>::value_ = eval();
  }
  void tally_parents() {
    for(vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
      (*iter)->tally();
    }
  }
  void tune(const double acceptance_rate) {
    for(vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
      (*iter)->tune(acceptance_rate);
    }
  }
  virtual void registerParents() = 0; // user must provide this function to make object aware of parents
  virtual T eval() = 0;  // user must provide this function to update object
};

template<typename T>
class MCMCStochastic : public MCMCSpecialized<T> {
protected:
  int iteration_;
  MCMCJumper<T> jumper_;
public:
  MCMCStochastic(const T& shape): MCMCSpecialized<T>(shape), iteration_(-1), jumper_(MCMCSpecialized<T>::value_) {}
  virtual double logp() const = 0;
  void jump(const int current_iteration) {
    // only jump if we hevn't already jumped yet
    if(iteration_ == current_iteration) {
      return;
    }
    ++iteration_;
    jumper_.jump();
  }
  void revert() {
    jumper_.revert();
  }
  void tune(const double acceptance_rate) {
    jumper_.tune(acceptance_rate);
  }
};

template<typename T>
class HyperPrior : public MCMCSpecialized<T> {
public:
  HyperPrior(const T& value) : MCMCSpecialized<T>(), MCMCSpecialized<T>::value_(value) {};
  double logp() const { return static_cast<double>(0); }
  void tally() {}
  void tally_parents() {}
};

template<typename T>
class Uniform : public MCMCStochastic<T> {
 private:
  const double lower_bound_;
  const double upper_bound_;
  boost::uniform_real<> rng_dist_;
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;
 public:
  Uniform(const double lower_bound, const double upper_bound, const T shape): MCMCStochastic<T>(shape),
                                                                              lower_bound_(lower_bound), upper_bound_(upper_bound),
                                                                              rng_dist_(lower_bound_,upper_bound_), rng_(MCMCStochastic<T>::generator_, rng_dist_) {    
    for(int i = 0; i < nrow(MCMCStochastic<T>::value_) * ncol(MCMCStochastic<T>::value_); i++) {
      MCMCStochastic<T>::value_[i] = rng_();
    }
    MCMCStochastic<T>::jumper_.setSD(sd());
  }
  double sd() {
    return (upper_bound_ - lower_bound_)/pow(12,0.5);
  }
  double logp() const {
    double ans(0);
    for(int i = 0; i < nrow(MCMCStochastic<T>::value_) * ncol(MCMCStochastic<T>::value_); i++) {
      if(MCMCStochastic<T>::value_[i] < lower_bound_ || MCMCStochastic<T>::value_[i] > upper_bound_) {
        return neg_inf;
      } else {
        ans += -log(upper_bound_ - lower_bound_);
      }
    }
    return ans;
  }
  void tally_parents() {}  // FIXME: will need this when upper/lower are alowed to be stocastics
};

template<typename T>
class LikelihoodFunctionObject {
  // for acceptace test
  base_generator_type generator_;
  boost::uniform_real<> uni_dist_;
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;
protected:
  const T& actual_values_;
  MCMCDeterministic<T>& forecaster_;
public:
  LikelihoodFunctionObject(const T& actual_values_, MCMCDeterministic<T>& forecaster): generator_(20u), uni_dist_(0,1), rng_(generator_, uni_dist_), actual_values_(actual_values_), forecaster_(forecaster) {}
  double rng() {
    return rng_();
  }
  virtual double logp() const = 0;

  void sample(int iterations, int burn, int thin) {
    double accepted(0);
    double rejected(0);

    for(int i = 0; i < iterations; i++) {
      double logp_old = logp();
      forecaster_.jump(i);
      double logp_new = logp();
      if(logp_new == neg_inf || log(rng()) > logp_new - logp_old) {
        forecaster_.revert();
        rejected+=1;
      } else {
        accepted+=1;
      }

      // tune every 50 during burn
      if(i % 50 == 0 && i < burn) {
        forecaster_.tune(accepted/(accepted + rejected));
        accepted = 0;
        rejected = 0;
      }

      // tune every 1000 during actual
      if(i % 1000 == 0) {
        forecaster_.tune(accepted/(accepted + rejected));
      }
      if(i > burn && i % thin == 0) {
        forecaster_.tally();
      }
    }
  }
};

template<typename T>
class NormalLikelihood : public LikelihoodFunctionObject<T> {
 private:
  const double tau_;
public:
  NormalLikelihood(const T& actual_values, MCMCDeterministic<T>& forecaster, const double tau): LikelihoodFunctionObject<T>(actual_values, forecaster), tau_(tau) {}

  double logp(const double value, const double mu, const double tau) const {
    return 0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(value-mu,2);
  }
  double logp() const {
    double ans(0);
    const T& sample = LikelihoodFunctionObject<T>::forecaster_.exposeValue();    
    for(int i = 0; i < LikelihoodFunctionObject<T>::actual_values_.n_elem; i++) {
      ans += logp(sample[i], LikelihoodFunctionObject<T>::actual_values_[i], tau_);
    }
    ans += LikelihoodFunctionObject<T>::forecaster_.logp();
    return ans;
  }
};

class EstimatedY : public MCMCDeterministic<mat> {
private:
  mat& X_;
  MCMCStochastic<vec>& b_;
public:
  EstimatedY(mat& X, MCMCStochastic<vec>& b): MCMCDeterministic<mat>(X * b.exposeValue()), X_(X), b_(b) {
    registerParents();
  }
  void registerParents() {
    parents_.push_back(&b_);
  }
  mat eval() {
    return X_ * b_.exposeValue();
  }
};

// global rng generators
gsl_rng* MCMCJumperBase::rng_source_ =  NULL;
base_generator_type MCMCObject::generator_;

int main() {
  const gsl_rng_type* T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  MCMCJumperBase::setupRNG(gsl_rng_alloc(T));

  const int N = 1000;
  mat X(N,2);
  mat y(N,1);

  base_generator_type sample_generator;
  normal_distribution<double> nd(0,2);
  boost::variate_generator<base_generator_type&, normal_distribution<double> > rng_(sample_generator, nd);

  for(int i = 0; i < N; i++) {
    y[i] = rng_();
    X(i,0) = 1;
    X(i,1) = rng_();
  }
  vec coefs;
  solve(coefs, X, y);
  Uniform<vec> B(-1.0,1.0, vec(2));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<mat> likelihood(y, obs_fcst, 100);
  int iterations = 1e5;
  likelihood.sample(iterations, 1e4, 4);
  const vector<vec>& coefs_hist(B.getHistory());

  vec avg_coefs(2);
  avg_coefs.fill(0);
  ofstream outfile;
  outfile.open ("coefs.csv");
  for(int i = 0; i < coefs_hist.size(); i++) {
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
