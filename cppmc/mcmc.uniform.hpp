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


  template<template<typename> class ArmaT>
  class Uniform : public MCMCStochastic<double,ArmaT> {
  private:
    MCMCSpecialized<double,Col>& lower_bound_;
    MCMCSpecialized<double,Col>& upper_bound_;
    boost::uniform_real<> rng_dist_;
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > rng_;
  public:
    Uniform(MCMCSpecialized<double,Col>& lower_bound, MCMCSpecialized<double,Col>&upper_bound, const ArmaT<double> shape):
      MCMCStochastic<double,ArmaT>(shape),
      lower_bound_(lower_bound), upper_bound_(upper_bound),
      rng_dist_(lower_bound_,upper_bound_), rng_(MCMCStochastic<double,ArmaT>::generator_, rng_dist_) {

      // FIXME: each value needs to use it's own lower/upper
      // set values
      for(size_t i = 0; i < MCMCStochastic<double,ArmaT>::size(); i++) {
	MCMCStochastic<double,ArmaT>::value_[i] = rng_();
      }
    }
    double calc_logp_self() const {
      double ans(0);
      const uint lower_size = lower_bound_.size();
      const uint upper_size = upper_bound_.size();
      for(size_t i = 0; i < MCMCStochastic<double,ArmaT>::size(); i++) {
	ans += uniform_logp(MCMCStochastic<double,ArmaT>::value_[i], lower_bound_[i % lower_size], upper_bound_[i % upper_size]);
      }
      return ans;
    }

    void registerParents() {
      MCMCStochastic<double,ArmaT>::parents_.push_back(lower_bound_);
      MCMCStochastic<double,ArmaT>::parents_.push_back(upper_bound_);
    }
  };
} // namespace CppMC
#endif // MCMC_UNIFORM_HPP
