///////////////////////////////////////////////////////////////////////////
// Copyright (C) 2009  Whit Armstrong                                    //
//                                                                       //
// This program is free software: you can redistribute it and/or modify  //
// it under the terms of the GNU General Public License as published by  //
// the Free Software Foundation, either version 3 of the License, or     //
// (at your option) any later version.                                   //
//                                                                       //
// This program is distributed in the hope that it will be useful,       //
// but WITHOUT ANY WARRANTY; without even the implied warranty of        //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         //
// GNU General Public License for more details.                          //
//                                                                       //
// You should have received a copy of the GNU General Public License     //
// along with this program.  If not, see <http://www.gnu.org/licenses/>. //
///////////////////////////////////////////////////////////////////////////

#ifndef MCMC_UNIFORM_HPP
#define MCMC_UNIFORM_HPP

#include <cppmc/mcmc.stochastic.hpp>

namespace CppMC {

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
} // namespace CppMC
#endif // MCMC_UNIFORM_HPP
