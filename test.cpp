#include <iostream>
#include <vector>
#include <limits>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/uniform.hpp>

using namespace boost;
using std::vector;
using std::cout;
using std::endl;
using boost::math::uniform;
typedef boost::minstd_rand base_generator_type;

const double neg_inf(std::numeric_limits<double>::min());

class MCMCObject {
private:
  unsigned int iterations_;
public:
  MCMCObject() : iterations_(0) {}
  void resetCount() { iterations_ = 0; }

  // virtuals
  virtual double logp(const double val) const = 0;
  virtual double logp() const = 0;
  virtual double getValue() const = 0;
  virtual void getChildren(vector<MCMCObject*>& holder) = 0;
  virtual void next() = 0;
  virtual void jump() = 0;
  virtual void reject() = 0;
};

class HyperPrior : public MCMCObject {
 private:
  double value_;
 public:
  HyperPrior(double value) : MCMCObject(), value_(value) {};
  double logp(const double val) const { return static_cast<double>(0); }
  double logp() const { return static_cast<double>(0); }
  double getValue() const { return value_; }
  // no children
  void getChildren(vector<MCMCObject*>& holder) {}
  // static? for duration of sim? do hypers change?
  void next() {}
  void jump() {}
  void reject() {}
};

class Likelihood {
private:
  MCMCObject& dist_;
  double* vals_;
  size_t n_;
  // for acceptace test
  typedef boost::minstd_rand base_generator_type;
  base_generator_type generator_;
  boost::uniform_real<> uni_dist_;
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;

  double logp() {
    double ans(0);
    for(size_t i = 0; i < n_; i++) {
      ans += dist_.logp(vals_[i]);
    }
    ans += dist_.logp();
    return ans;
  }
public:
  Likelihood(MCMCObject& dist, double* vals, size_t n): dist_(dist), vals_(vals), n_(n), generator_(42u), uni_dist_(0,1), rng_(generator_, uni_dist_) {}
  void sample(size_t iterations, size_t burn, size_t thin) {
    for(size_t i = 0; i < iterations; i++) {
      double logp_old = logp();
      dist_.jump();
      double logp_new = logp();
      if(log(rng_()) > logp_new - logp_old) {
	dist_.reject();
      }
    }
  }
};

class Uniform : public MCMCObject {
 private:
  // This is a typedef for a random number generator.
  // Try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
  typedef boost::minstd_rand base_generator_type;
  base_generator_type generator_;
  boost::uniform_real<> current_dist_;
  boost::uniform_real<> old_dist_;
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;

  normal_distribution<double> jumper_dist_;
  boost::variate_generator<base_generator_type&, normal_distribution<double> > jumper_;

  MCMCObject& lower_bound_;
  MCMCObject& upper_bound_;
  uniform rng_source;
  double value_;
 public:
  Uniform(MCMCObject& lower_bound, MCMCObject& upper_bound) :
    MCMCObject(),
    lower_bound_(lower_bound), upper_bound_(upper_bound), generator_(42u), current_dist_(lower_bound.getValue(),upper_bound.getValue()), rng_(generator_, current_dist_) {};

  double logp(const double value) const {
    return (value_ < lower_bound_.getValue() || value_ > upper_bound_.getValue()) ? neg_inf : -log(upper_bound_.getValue()-lower_bound_.getValue());
  }
  double logp() const {
    return  logp(value_) + lower_bound_.logp() + upper_bound_.logp();
  }
  double getValue() const { return value_; }
  void getChildren(vector<MCMCObject*>& holder) {
    holder.push_back(&lower_bound_);
    holder.push_back(&upper_bound_);
  }
  void next() {
    value_ = rng_();
  }
  void jump() {
    value_
  }
  void reject() {}
};


int main() {
  const int N = 100;
  double sample_data[N];

  typedef boost::minstd_rand base_generator_type;
  base_generator_type generator_;
  normal_distribution<double> norm_dist_(0,1);
  boost::variate_generator<base_generator_type&, normal_distribution<double> > rng_(generator_, norm_dist_);

  for(int i = 0; i < 10; i++) {
    sample_data[i] = rng_();
    cout << sample_data[i] << endl;
  }

  HyperPrior lower(0);
  HyperPrior upper(1);
  Uniform beta(lower, upper);

  return 1;
}
