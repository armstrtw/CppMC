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

  template<typename T>
  class Normal : public MCMCStochastic<T> {
  private:
    const double mu_;
    const double tau_;
    boost::normal_distribution<double> rng_dist_;
    boost::variate_generator<base_generator_type&, boost::normal_distribution<double> > rng_;
  public:
    Normal(const double mu, const double tau, const T shape): MCMCStochastic<T>(shape),
                                                              mu_(mu), tau_(tau),
                                                              rng_dist_(mu_, tau_), rng_(MCMCStochastic<T>::generator_, rng_dist_) {
      for(size_t i = 0; i < nrow(MCMCStochastic<T>::value_) * ncol(MCMCStochastic<T>::value_); i++) {
	MCMCStochastic<T>::value_[i] = rng_();
      }
      MCMCStochastic<T>::jumper_.setSD(sd());
    }
    double sd() {
      return sqrt(tau_);
    }
    double calc_logp_self() const {
      double ans(0);
      for(size_t i = 0; i < nrow(MCMCStochastic<T>::value_) * ncol(MCMCStochastic<T>::value_); i++) {
	ans += normal_logp(MCMCStochastic<T>::value_[i], mu_, tau_);
      }
      return ans;
    }
    void tally_parents() {}
  };
} // namespace CppMC
#endif // MCMC_NORMAL_HPP
