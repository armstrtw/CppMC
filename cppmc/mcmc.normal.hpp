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

#ifndef MCMC_NORMAL_HPP
#define MCMC_NORMAL_HPP

#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>
#include <cppmc/mcmc.stochastic.hpp>

namespace CppMC {

  template<typename DataT,
           template<typename> class ArmaT>
  class Normal : public MCMCStochastic<DataT,ArmaT> {
  private:
    const double mu_;
    const double tau_;
    boost::normal_distribution<double> rng_dist_;
    boost::variate_generator<base_generator_type&, boost::normal_distribution<double> > rng_;
  public:
    Normal(const double mu, const double standard_deviation, const ArmaT<DataT> shape): MCMCStochastic<DataT,ArmaT>(shape),
                                                                                        mu_(mu), tau_(MCMCObject::sd_to_tau(standard_deviation)),
                                                                                        rng_dist_(mu_, standard_deviation), rng_(MCMCStochastic<DataT,ArmaT>::generator_, rng_dist_) {
      for(size_t i = 0; i < MCMCStochastic<DataT,ArmaT>::size(); i++) {
	MCMCStochastic<DataT,ArmaT>::value_[i] = rng_();
      }
      MCMCStochastic<DataT,ArmaT>::jumper_.setSD(sd());
    }
    double sd() const {
      return MCMCObject::tau_to_sd(tau_);
    }
    double calc_logp_self() const {
      double ans(0);
      for(size_t i = 0; i < MCMCStochastic<DataT,ArmaT>::size(); i++) {
	ans += normal_logp(MCMCStochastic<DataT,ArmaT>::value_[i], mu_, tau_);
      }
      return ans;
    }
    void tally_parents() {}

    // need to define when mu and sd are allowed to be MCMC objects
    void registerParents() {}
  };
} // namespace CppMC
#endif // MCMC_NORMAL_HPP
