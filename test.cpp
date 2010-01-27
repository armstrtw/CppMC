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

const double neg_inf(-std::numeric_limits<double>::infinity());

//#define debug

class MCMCJumper {
 private:
  double mean_;
  double sd_;
  double old_mean_;
  double old_sd_;
  static gsl_rng* rng_source_;
 public:
  MCMCJumper(const double mean, const double sd): mean_(mean), sd_(sd), old_mean_(0), old_sd_(0) {}
  static void setupRNG(gsl_rng* rng) {
    rng_source_ = rng;
  }
  double getSD() const { return sd_; }
  void setSD(const double sd) { sd_ = sd; }
  double rng() {
#ifdef debug
    cout << "mean:sd: " << mean_ << " " << sd_ << endl;
#endif
    return mean_ + gsl_ran_gaussian(rng_source_, sd_);
  }
  void jump(const double new_mean, const double new_sd) {
#ifdef debug
    cout << "jumping" << endl;
#endif
    old_mean_ = mean_;
    old_sd_ = sd_;
    mean_ = new_mean;
    sd_ = new_sd;
  }
  void revert() {
#ifdef debug
    cout << "reverting" << endl;
#endif
    mean_ = old_mean_;
    sd_ = old_sd_;
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
public:
  MCMCSpecialized(): MCMCObject() {}
  // does a stub for (virtual double logp() const = 0;) need to be defined here?
  virtual T getValue() const = 0;
  // cannot be overloaded: virtual const T& getValue() const = 0;
};

template<typename T>
class MCMCDeterministic : public MCMCSpecialized<T> {
public:
  virtual double logp() const = 0; // must return logp of children
  virtual void jump(int current_iteration) = 0; // must jump children
  virtual void reject() = 0; // must reject children
  virtual T getValue() const = 0; // pass through
};

template<typename T>
class MCMCStochastic : public MCMCSpecialized<T> {
protected:
  int iteration_;
  T value_;
  MCMCJumper jumper_;
public:
  MCMCStochastic() : iteration_(-1), jumper_(value_, 1.0) {}
  // virtuals
  T getValue() const { return value_; }
  virtual double logp(const double val) const = 0;
  virtual double logp() const = 0;
  void next_rng() {
    value_ = jumper_.rng();
  }
  void jump(const int current_iteration) {
    // only jump if we hevn't already jumped yet
    if(iteration_ == current_iteration) {
      return;
    }
    //cout << "jump" << endl;
    // increment iterations
    ++iteration_;

    // set rng mean to current value
    // preserve sd for now
    jumper_.jump(value_, jumper_.getSD());

    // update rng of value_ based on jumped distribution
    next_rng();
    //cout << "new value:" << value_ << endl;
  }
  void reject() {
    jumper_.revert();
    // update rng of value_ based on reverted distribution
    next_rng();
  }
};

template<typename T>
class HyperPrior : public MCMCSpecialized<T> {
protected:
  const T value_;
public:
  HyperPrior(const T& value) : MCMCSpecialized<T>(), value_(value) {};
  double logp() const { return static_cast<double>(0); }
  virtual T getValue() { return value_; }
};

// template<typename T>
// class Likelihood {
// private:
//   MCMCStochastic<T>& dist_;
//   T actual_obs_;
//   MCMCDeterministic<T>& forecaster_;
//   size_t n_;

//   // for acceptace test
//   base_generator_type generator_;
//   boost::uniform_real<> uni_dist_;
//   boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;

// public:
//   Likelihood(MCMCStochastic<T>& dist, const T& actual_obs, MCMCDeterministic<T>& forecaster): dist_(dist), actual_obs_(actual_obs), forecaster_(forecaster), generator_(42u), uni_dist_(0,1), rng_(generator_, uni_dist_), n_(actual_obs.n_elem) {}

//   void sample(size_t iterations, size_t burn, size_t thin) {
//     for(size_t i = 0; i < iterations; i++) {
//       double logp_old = logp();
//       forecaster_.jump(i);
//       double logp_new = logp();
//       if(log(rng_()) > logp_new - logp_old) {
// 	forecaster_.reject();
//       }
//     }
//   }

//   double logp() {
//     double ans(0);
//     T fcst = forecaster_.getValue();
//     for(size_t i = 0; i < n_; i++) {
//       ans += dist_.logp(fcst[i]);
//     }
//     //cout << "obs logp: " << ans << endl;
//     //cout << "forecaster logp: " << forecaster_.logp() << endl;
//     ans += forecaster_.logp();

//     return ans;
//   }
// };

template<typename T>
class Uniform : public MCMCStochastic<T> {
 private:
  const double lower_bound_;
  const double upper_bound_;
  boost::uniform_real<> rng_dist_;
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;
 public:
  Uniform(const double lower_bound, const double upper_bound) :
    MCMCStochastic<T>(),
    lower_bound_(lower_bound), upper_bound_(upper_bound), rng_dist_(lower_bound_,upper_bound_), rng_(MCMCStochastic<T>::generator_, rng_dist_)
  {
    MCMCStochastic<T>::jumper_.setSD(getSD());
    MCMCStochastic<T>::value_= rng_();
    cout << "initial value: " << MCMCStochastic<T>::value_ << endl;
  };
  double getSD() {
    return (upper_bound_ - lower_bound_)/pow(12,0.5);
  }
  double logp(const double value) const {
     if(MCMCStochastic<T>::value_ < lower_bound_ || MCMCStochastic<T>::value_ > upper_bound_) {
#ifdef debug
       cout << "oob" << endl;
#endif
     }
    return (MCMCStochastic<T>::value_ < lower_bound_ || MCMCStochastic<T>::value_ > upper_bound_) ? neg_inf : -log(upper_bound_ - lower_bound_);
  }
  double logp() const {
    return  logp(MCMCStochastic<T>::value_);
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
    // logistic log, reuse later: return log(tau_) - tau_ * (value - mu_) - 2.0*log(1.0 + exp(-tau_ * (value-mu_)));
    return - 0.5 * tau * pow(value-mu,2) + 0.5*log(0.5*tau/arma::math::pi());
  }
  double logp() const {
    return  logp(MCMCStochastic<T>::value_, mu_, tau_);
   }
 };

class EstimatedY : public MCMCDeterministic<mat> {
private:
  const mat& vals_;
  MCMCStochastic<double>& alpha_;
  MCMCStochastic<double>& beta_;
public:
  EstimatedY(const mat& vals, MCMCStochastic<double>& alpha, MCMCStochastic<double>& beta): vals_(vals), alpha_(alpha), beta_(beta) {}

  mat getValue() const {
#ifdef debug
    cout << "alpha:beta: " << alpha_.getValue() <<":"<< beta_.getValue() << endl;
#endif
    return alpha_.getValue() + beta_.getValue() * vals_;
  }
  double logp() const { return alpha_.logp() + beta_.logp(); }
  void jump(int current_iteration) { alpha_.jump(current_iteration); beta_.jump(current_iteration); }
  void reject() { alpha_.reject(); beta_.reject(); }
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
    T sample = LikelihoodFunctionObject<T>::forecaster_.getValue();
    double ans(0);
    for(int i = 0; i < LikelihoodFunctionObject<T>::actual_values_.n_elem; i++) {
      ans += logp(sample[i], LikelihoodFunctionObject<T>::actual_values_[i], tau_);
    }
    ans += LikelihoodFunctionObject<T>::forecaster_.logp();
    return ans;
  }
  void sample(size_t iterations, size_t burn, size_t thin) {
    for(size_t i = 0; i < iterations; i++) {
      double logp_old = logp();
      LikelihoodFunctionObject<T>::forecaster_.jump(i);
      double logp_new = logp();
#ifdef debug
      cout << "old, new: " << logp_old << " : " << logp_new << endl;
#endif
      if(logp_new == neg_inf || log(LikelihoodFunctionObject<T>::rng()) > logp_new - logp_old) {
	LikelihoodFunctionObject<T>::forecaster_.reject();
      }
    }
  }
};


// global rng generators
gsl_rng* MCMCJumper::rng_source_ =  NULL;
base_generator_type MCMCObject::generator_;

int main() {
  const gsl_rng_type* T;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  MCMCJumper::setupRNG(gsl_rng_alloc(T));

  const int N = 1000;
  mat X(N,2);
  mat y(N,1);

  base_generator_type sample_generator;
  normal_distribution<double> nd(0,1);
  boost::variate_generator<base_generator_type&, normal_distribution<double> > rng_(sample_generator, nd);

  for(int i = 0; i < N; i++) {
    y[i] = rng_();
    X[i,1] = rng_();
    X[i,2] = rng_();
  }
  vec coefs;
  solve(coefs, X, y);
  cout << "lm coefs:" << endl;
  cout << coefs << endl;
  Uniform<double> alpha(-1,1);
  Uniform<double> beta(-1,1);
  EstimatedY obs_fcst(X, alpha, beta);
  NormalLikelihood<mat> likelihood(y, obs_fcst, 0.0001);
  likelihood.sample(1e2, 10, 2);
  return 1;
}
