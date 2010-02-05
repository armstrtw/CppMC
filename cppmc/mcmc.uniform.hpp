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

  template<typename DataT,
           template<typename> class ArmaT>
  class Uniform : public MCMCStochastic<DataT,ArmaT> {
  private:
    const double lower_bound_;
    const double upper_bound_;
    boost::uniform_real<> rng_dist_;
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;
  public:
    Uniform(const double lower_bound, const double upper_bound, const ArmaT<DataT> shape): MCMCStochastic<DataT,ArmaT>(shape),
                                                                                     lower_bound_(lower_bound), upper_bound_(upper_bound),
                                                                                     rng_dist_(lower_bound_,upper_bound_), rng_(MCMCStochastic<DataT,ArmaT>::generator_, rng_dist_) {
      for(size_t i = 0; i < MCMCStochastic<DataT,ArmaT>::size(); i++) {
	MCMCStochastic<DataT,ArmaT>::value_[i] = rng_();
      }
      MCMCStochastic<DataT,ArmaT>::jumper_.setSD(sd());
    }
    double calc_logp_self() const {
      double ans(0);
      for(size_t i = 0; i < MCMCStochastic<DataT,ArmaT>::size(); i++) {
	ans += uniform_logp(MCMCStochastic<DataT,ArmaT>::value_[i], lower_bound_, upper_bound_);
      }
      return ans;
    }

    void registerParents() {
      // only when lower_bound_ and upper_bound_ are declared as MCMCobjects
      //MCMCStochastic<DataT,ArmaT>::parents_.push_back(lower_bound_);
      //MCMCStochastic<DataT,ArmaT>::parents_.push_back(upper_bound_);
    }
    double sd() const {
      return (upper_bound_ - lower_bound_)/pow(12,0.5);
    }
  };
} // namespace CppMC
#endif // MCMC_UNIFORM_HPP
