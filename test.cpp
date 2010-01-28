#include <iostream>
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
using std::cout;
using std::endl;
using boost::math::uniform;
typedef boost::minstd_rand base_generator_type;
//#define DEBUG

const double neg_inf(-std::numeric_limits<double>::infinity());

inline int nrow(vec v) {
  return v.n_rows;
}

inline int ncol(vec v) {
  return v.n_cols;
}

inline int nrow(mat m) {
  return m.n_rows;
}

inline int ncol(mat m) {
  return m.n_cols;
}

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
  T mean_;
  T sd_;
  T old_mean_;
  T& value_;
  double scale_;
  const int n_;
  void drawRNG() {
    for(int i = 0; i < n_; i++) {
      value_[i] = mean_[i] + sd_[i] * gsl_ran_gaussian(rng_source_, 1.0);
    }
  }
 public:
  MCMCJumper(T& value): mean_(value), sd_(abs(value)), old_mean_(value), value_(value), n_(nrow(value) *ncol(value)), scale_(1) {}
  void setScale(const double scale) {
    scale_ = scale;
  }
  void jump() {
    old_mean_ = mean_;
    mean_ = value_;
    //cout << "mean" << endl << mean_;
    //cout << "value" << endl << value_;
    drawRNG();
  }
  void revert() {
    mean_ = old_mean_;
    drawRNG();
  }

  void tune(const double acceptance_rate) {
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
    if(acceptance_rate > .95) {
      scale_ *= 10;
      return;
    }
    if(acceptance_rate > .75) {
      scale_ *= 2;
      return;
    }
    if(acceptance_rate < .5) {
      scale_ *= 1.1;
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

// can't have value_ here b/c deterministics actually don't store their value
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
    //cout << MCMCSpecialized<T>::value_ << endl;
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
    lower_bound_(lower_bound), upper_bound_(upper_bound), rng_dist_(lower_bound_,upper_bound_), rng_(MCMCStochastic<T>::generator_, rng_dist_) {
    MCMCStochastic<T>::jumper_.setScale(getSD()/2);
  }
  double getSD() {
    return (upper_bound_ - lower_bound_)/pow(12,0.5);
  }
  double logp() const {
    double ans(0);
    for(int i = 0; i < nrow(MCMCStochastic<T>::value_) * ncol(MCMCStochastic<T>::value_); i++) {
      ans += (MCMCStochastic<T>::value_[i] < lower_bound_ || MCMCStochastic<T>::value_[i] > upper_bound_) ? neg_inf : -log(upper_bound_ - lower_bound_);
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
  LikelihoodFunctionObject(const T& actual_values_, MCMCDeterministic<T>& forecaster): generator_(50u), uni_dist_(0,1), rng_(generator_, uni_dist_), actual_values_(actual_values_), forecaster_(forecaster) {}
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
#ifdef DEBUG
      cout << "old, new: " << logp_old << " : " << logp_new << endl;
#endif
      if(logp_new == neg_inf || log(rng()) > logp_new - logp_old) {
        //cout << "revert" << endl;
        forecaster_.revert();
        ++rejected;
      } else {
        ++accepted;
      }

      // tune every 20
      if(i < burn && i % 20 == 0) {
        forecaster_.tune(accepted/accepted + rejected);
        accepted = 0;
        rejected = 0;
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

  const int N = 10;
  mat X(N,2);
  mat y(N,1);

  base_generator_type sample_generator;
  normal_distribution<double> nd(0,1);
  boost::variate_generator<base_generator_type&, normal_distribution<double> > rng_(sample_generator, nd);

  for(int i = 0; i < N; i++) {
    y[i] = rng_();
    X(i,0) = 1;
    X(i,1) = rng_();
  }
  vec coefs;
  solve(coefs, X, y);
  cout << "lm coefs:" << endl;
  cout << coefs << endl;
  Uniform<vec> B(-1.0,1.0, vec(2));
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<mat> likelihood(y, obs_fcst, 0.0001);
  likelihood.sample(1e6, 1e3, 2);
  const vector<vec>& coefs_hist(B.getHistory());

  vec avg_coefs(2);
  avg_coefs.fill(0);
  for(int i = 0; i < coefs_hist.size(); i++) {
    //cout << coefs_hist[i][0] << " " << coefs_hist[i][1] << endl;
    avg_coefs += coefs_hist[i];
  }
  avg_coefs /= coefs_hist.size();

  cout << "colected " << coefs_hist.size() << " samples." << endl;
  cout << "lm coefs" << endl << coefs << endl;
  cout << "avg_coefs" << endl << avg_coefs << endl;
  return 1;
}
