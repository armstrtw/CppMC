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
#define DEBUG

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
  T old_sd_;
  T& value_;
  const int n_;
  void drawRNG() {
    for(int i = 0; i < n_; i++) {
      value_[i] = mean_[i] + gsl_ran_gaussian(rng_source_, sd_[i]);
    }
  }
 public:
  MCMCJumper(T& value): mean_(value), sd_(ones(nrow(value) *ncol(value))), old_mean_(value), old_sd_(value), value_(value), n_(nrow(value) *ncol(value)) { sd_.fill(0.1); }
  const T& getSD() const { return sd_; }
  void setSD(const T& sd) { sd_ = sd; }
  void jump() {
    old_mean_ = mean_;
    old_sd_ = sd_;
    mean_ = value_;
    //sd_ = new_sd;
    drawRNG();
  }
  void revert() {
    mean_ = old_mean_;
    sd_ = old_sd_;
    drawRNG();
  }
};

class MCMCObject {
protected:
  static base_generator_type generator_;
public:
  MCMCObject() {}
  // every object must have a way to calc logp
  virtual double logp() const = 0;
};

// can't have value_ here b/c deterministics actually don't store their value
template<typename T>
class MCMCSpecialized : public MCMCObject {
protected:
  T value_;
public:
  MCMCSpecialized(const T& shape): MCMCObject(), value_(shape) {}
  const T& exposeValue() const {
    return value_;
  }
};

template<typename T>
class MCMCDeterministic : public MCMCSpecialized<T> {
public:
  MCMCDeterministic(const T& initial_value): MCMCSpecialized<T>(initial_value) {}
  virtual double logp() const = 0; // must return logp of children
  virtual void jump(int current_iteration) = 0; // must jump children, and update self
  virtual void revert() = 0; // must revert children, and update self
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
};

template<typename T>
class HyperPrior : public MCMCSpecialized<T> {
public:
  HyperPrior(const T& value) : MCMCSpecialized<T>(), MCMCSpecialized<T>::value_(value) {};
  double logp() const { return static_cast<double>(0); }
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
    lower_bound_(lower_bound), upper_bound_(upper_bound), rng_dist_(lower_bound_,upper_bound_), rng_(MCMCStochastic<T>::generator_, rng_dist_)
  {
    cout << "shape passed in: " << endl << shape << endl;
    cout << "initial value: " << endl << MCMCStochastic<T>::value_ << endl;
  };
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
};

template<typename T>
class Normal : public MCMCStochastic<T> {
private:
  const double mu_;
  const double tau_;
  normal_distribution<double> rng_dist_;
  boost::variate_generator<base_generator_type&, normal_distribution<double> > rng_;
public:
  Normal(const double mu, const double tau) :
    MCMCStochastic<T>(),
    mu_(mu), tau_(tau), rng_dist_(mu_,pow(tau_,2)), rng_(MCMCStochastic<T>::generator_, rng_dist_)
  {
    MCMCStochastic<T>::jumper_.setSD(getSD());
    MCMCStochastic<T>::value_= rng_();
  }
  double getSD() {
    return sqrt(tau_);
  }
  double logp(const double value, const double mu, const double tau) const {
    return - 0.5 * tau * pow(value-mu,2) + 0.5*log(0.5*tau/arma::math::pi());
  }
  double logp() const {
    return  logp(MCMCStochastic<T>::value_, mu_, tau_);
   }
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
};

template<typename T>
class NormalLikelihood : public LikelihoodFunctionObject<T> {
 private:
  const double tau_;
public:
  NormalLikelihood(const T& actual_values, MCMCDeterministic<T>& forecaster, const double tau): LikelihoodFunctionObject<T>(actual_values, forecaster), tau_(tau) {}
  double logp(const double value, const double mu, const double tau) const {
    return - 0.5 * tau * pow(value-mu,2) + 0.5*log(0.5*tau/arma::math::pi());
  }
  double logp() const {
    const T& sample = LikelihoodFunctionObject<T>::forecaster_.exposeValue();
    double ans(0);
    for(int i = 0; i < LikelihoodFunctionObject<T>::actual_values_.n_elem; i++) {
      //cout << "sample[i]" << sample[i] << endl;
      ans += logp(sample[i], LikelihoodFunctionObject<T>::actual_values_[i], tau_);
    }
    ans += LikelihoodFunctionObject<T>::forecaster_.logp();
    return ans;
  }
  void sample(int iterations, int burn, int thin) {
    for(int i = 0; i < iterations; i++) {
      double logp_old = logp();
      LikelihoodFunctionObject<T>::forecaster_.jump(i);
      double logp_new = logp();
#ifdef DEBUG
      cout << "old, new: " << logp_old << " : " << logp_new << endl;
#endif
      if(logp_new == neg_inf || log(LikelihoodFunctionObject<T>::rng()) > logp_new - logp_old) {
        cout << "revert" << endl;
	LikelihoodFunctionObject<T>::forecaster_.revert();
      }
    }
  }
};

class EstimatedY : public MCMCDeterministic<mat> {
private:
  mat& X_;
  MCMCStochastic<vec>& b_;
public:
  EstimatedY(mat& X, MCMCStochastic<vec>& b): MCMCDeterministic<mat>(X * b.exposeValue()), X_(X), b_(b) {
    cout << "X:" << endl << X_;
    //cout << "b:" << endl << b_;
    cout << "value_:" << endl << value_;
  }
  double logp() const { return b_.logp(); }
  void jump(int current_iteration) {
    b_.jump(current_iteration);
    value_ = X_ * b_.exposeValue();
  }
  void revert() {
    b_.revert();
    value_ = X_ * b_.exposeValue();
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

  const int N = 100;
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
  //cout << B.exposeValue() << endl;
  EstimatedY obs_fcst(X, B);
  NormalLikelihood<mat> likelihood(y, obs_fcst, 0.0001);
  likelihood.sample(1e4, 10, 2);
  return 1;
}
