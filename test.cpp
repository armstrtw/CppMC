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

const double neg_inf(std::numeric_limits<double>::min());

class MCMCJumper {
 private:
  double mean_;
  double sd_;
  double old_mean_;
  double old_sd_;
  const gsl_rng_type* T_;
  gsl_rng* rng_source_;
 public:
  ~MCMCJumper() {
    gsl_rng_free (rng_source_);
  }
  MCMCJumper(const double mean, const double sd): mean_(mean), sd_(sd), old_mean_(0), old_sd_(0) {
    gsl_rng_env_setup();
    T_ = gsl_rng_default;
    rng_source_ = gsl_rng_alloc (T_);
  }
  double rng() {
    return mean_ + gsl_ran_gaussian(rng_source_, sd_);
  }
  void jump(const double new_mean, const double new_sd) {
    old_mean_ = mean_;
    old_sd_ = sd_;
    mean_ = new_mean;
    sd_ = new_sd;
  }
  void revert() {
    mean_ = old_mean_;
    sd_ = old_sd_;
  }
};

class MCMCObject {
private:
  virtual double logp() const = 0;
};

template<typename T>
class MCMCDeterministic : public MCMCObject {
public:
  virtual double logp() const = 0; // must return logp of children
  virtual void jump(int current_iteration) = 0; // must jump children
  virtual void reject() = 0; // must reject children
  T getValue();
};

class MCMCStochastic : public MCMCObject {
private:
  int iteration_;
protected:
  MCMCJumper jumper_;
  double value_;
public:
  MCMCStochastic() : iteration_(-1), jumper_(value_, 1.0) {}
  // virtuals
  virtual double logp(const double val) const = 0;
  virtual double logp() const = 0;
  double getValue() const { return value_; }
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
    // always pass sd = 1.0 for now
    jumper_.jump(value_, 1.0);

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

class HyperPrior : public MCMCObject {
 private:
  double value_;
 public:
  HyperPrior(double value) : MCMCObject(), value_(value) {};
  double logp() const { return static_cast<double>(0); }
};

template<typename T>
class Likelihood {
private:
  MCMCStochastic& dist_;
  T actual_obs_;
  MCMCDeterministic<T>& forecaster_;
  size_t n_;

  // for acceptace test
  typedef boost::minstd_rand base_generatorT;
  base_generatorT generator_;
  boost::uniform_real<> uni_dist_;
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;

  double logp() {
    double ans(0);
    for(size_t i = 0; i < n_; i++) {
      ans += dist_.logp(actual_obs_[i]);
    }
    //cout << "obs logp: " << ans << endl;
    ans += dist_.logp();

    return ans;
  }
public:
  Likelihood(MCMCStochastic& dist, const T& actual_obs, MCMCDeterministic<T>& forecaster): dist_(dist), actual_obs_(actual_obs), forecaster_(forecaster), generator_(42u), uni_dist_(0,1), rng_(generator_, uni_dist_), n_(actual_obs.n_elem) {}

  void sample(size_t iterations, size_t burn, size_t thin) {
    for(size_t i = 0; i < iterations; i++) {
      double logp_old = logp();
      forecaster_.jump(i);
      double logp_new = logp();
      if(log(rng_()) > logp_new - logp_old) {
	forecaster_.reject();
      }
    }
  }
};

class Uniform : public MCMCStochastic {
 private:
  // This is a typedef for a random number generator.
  // Try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
  typedef boost::minstd_rand base_generator_type;
  base_generator_type generator_;
  boost::uniform_real<> rng_dist_;
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;

  const double lower_bound_;
  const double upper_bound_;
 public:
  Uniform(const double lower_bound, const double upper_bound) :
    MCMCStochastic(),
    lower_bound_(lower_bound), upper_bound_(upper_bound), generator_(42u), rng_dist_(lower_bound,upper_bound), rng_(generator_, rng_dist_)
  {
    value_= rng_();
    cout << "initial value: " << value_ << endl;
  };

  double logp(const double value) const {
    return (value_ < lower_bound_ || value_ > upper_bound_) ? neg_inf : -log(upper_bound_ - lower_bound_);
  }
  double logp() const {
    return  logp(value_);
  }
};

class Normal : public MCMCStochastic {
 private:
  // This is a typedef for a random number generator.
  // Try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
  typedef boost::minstd_rand base_generator_type;
  base_generator_type generator_;
  normal_distribution<double> rng_dist_;
  boost::variate_generator<base_generator_type&, normal_distribution<double> > rng_;

  const double mu_;
  const double tau_;
public:
  Normal(const double mu, const double tau) :
    MCMCStochastic(),
    mu_(mu), tau_(tau), generator_(42u), rng_dist_(mu_,pow(tau_,2)), rng_(generator_, rng_dist_) { value_= rng_(); };

  double logp(const double value) const {
    // logistic log, reuse later: return log(tau_) - tau_ * (value - mu_) - 2.0*log(1.0 + exp(-tau_ * (value-mu_)));
    return - 0.5 * tau_ * pow(value-mu_,2) + 0.5*log(0.5*tau_/arma::math::pi());
  }
  double logp() const {
    return  logp(value_);
  }
};

class EstimatedY : public MCMCDeterministic<mat> {
private:
  MCMCStochastic& alpha_;
  MCMCStochastic& beta_;
  const mat& vals_;
public:
  EstimatedY(const mat& vals, MCMCStochastic& alpha, MCMCStochastic& beta): vals_(vals), alpha_(alpha), beta_(beta) {}
  mat getValue() { return alpha_.getValue() + beta_.getValue() * vals_; }
  double logp() const { return alpha_.logp() + beta_.logp(); }
  void jump(int current_iteration) { alpha_.jump(current_iteration); beta_.jump(current_iteration); }
  void reject() { alpha_.reject(); beta_.reject(); }
};

int main() {
  const int N = 30;
  mat X(N,1);
  mat y(N,1);

  typedef boost::minstd_rand base_generator_type;
  base_generator_type generator_;
  normal_distribution<double> norm_dist_(0,1);
  boost::variate_generator<base_generator_type&, normal_distribution<double> > rng_(generator_, norm_dist_);

  for(int i = 0; i < 100; i++) {
    y[i] = rng_();
    X[i] = rng_();
  }
  Uniform alpha(-1,1);
  Uniform beta(-1,1);
  EstimatedY obs_fcst(X, alpha, beta);
  Normal y_hat(0, 0.0001);
  Likelihood<mat> likelihood(y_hat, y, obs_fcst);
  likelihood.sample(1e3, 1e2, 10);
  return 1;
}
