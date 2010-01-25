#include <iostream>
#include <vector>
#include <limits>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/math/distributions/uniform.hpp>
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
  double logp() const { return static_cast<double>(0); }
  double getValue() const { return value_; }
  // no children
  void getChildren(vector<MCMCObject*>& holder) {}
  // static? for duration of sim? do hypers change?
  void next() {}
  void jump() {}
  void reject() {}
};

class Uniform : public MCMCObject {
 private:
  // This is a typedef for a random number generator.
  // Try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
  typedef boost::minstd_rand base_generator_type;
  base_generator_type generator_;
  boost::uniform_real<> uni_dist_;
  boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;

  MCMCObject& lower_bound_;
  MCMCObject& upper_bound_;
  uniform rng_source;
  double value_;
 public:
  Uniform(MCMCObject& lower_bound, MCMCObject& upper_bound) :
    MCMCObject(),
    lower_bound_(lower_bound), upper_bound_(upper_bound), generator_(42u), uni_dist_(lower_bound.getValue(),upper_bound.getValue()), rng_(generator_, uni_dist_) {};

  double logp() const {
    double self_logp = (value_ < lower_bound_.getValue() || value_ > upper_bound_.getValue()) ? neg_inf : -log(upper_bound_.getValue()-lower_bound_.getValue());
    return  self_logp + lower_bound_.logp() + upper_bound_.logp();
  }
  double getValue() const { return value_; }
  void getChildren(vector<MCMCObject*>& holder) {
    holder.push_back(&lower_bound_);
    holder.push_back(&upper_bound_);
  }
  void next() {
    value_ = rng_();
  }
  void jump() {}
  void reject() {}
};


int main() {
  HyperPrior lower(0);
  HyperPrior upper(10);
  Uniform uni(lower, upper);

  for(int i = 0; i < 100; i++) {
    uni.next();
    cout << "val/logp: " <<  uni.getValue() << ":" << uni.logp() << endl;
  }
  return 1;
}
