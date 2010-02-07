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
           template<typename> class ArmaPriorT,
           template<typename> class ArmaSelfT>
  class Normal : public MCMCStochastic<DataT,ArmaSelfT> {
  private:
    MCMCSpecialized<DataT,ArmaPriorT>& mu_;
    MCMCSpecialized<DataT,ArmaPriorT>& tau_;

    boost::normal_distribution<double> rng_dist_;
    boost::variate_generator<base_generator_type&, boost::normal_distribution<double> > rng_;
  public:
    Normal(MCMCSpecialized<DataT,ArmaPriorT>&mu, MCMCSpecialized<DataT,ArmaPriorT>& standard_deviation, const ArmaSelfT<DataT> shape):
      MCMCStochastic<DataT,ArmaSelfT>(shape),
      mu_(mu), tau_(MCMCObject::sd_to_tau(standard_deviation)),
      rng_dist_(mu_, standard_deviation), rng_(MCMCStochastic<DataT,ArmaSelfT>::generator_, rng_dist_) {

      for(size_t i = 0; i < MCMCStochastic<DataT,ArmaSelfT>::size(); i++) {
	MCMCStochastic<DataT,ArmaSelfT>::value_[i] = rng_();
      }
      MCMCStochastic<DataT,ArmaSelfT>::jumper_.setSD(sd());
    }
    ArmaSelfT<DataT> sd() const {
      ArmaSelfT<DataT> ans(tau_.nrow(),tau_.ncol());
      for(uint i = 0; i < tau_.size(); i++) {
        ans[i] = MCMCObject::tau_to_sd(tau_[i]);
      }
      return ans;
    }
    double calc_logp_self() const {
      double ans(0);
      for(size_t i = 0; i < MCMCStochastic<DataT,ArmaSelfT>::size(); i++) {
	ans += normal_logp(MCMCStochastic<DataT,ArmaSelfT>::value_[i], mu_[i], tau_[i]);
      }
      return ans;
    }
    void tally_parents() {}

    // need to define when mu and sd are allowed to be MCMC objects
    void registerParents() {
      MCMCStochastic<DataT,ArmaSelfT>::parents_.push_back(&mu_);
      MCMCStochastic<DataT,ArmaSelfT>::parents_.push_back(&tau_);
    }
  };
} // namespace CppMC
#endif // MCMC_NORMAL_HPP
