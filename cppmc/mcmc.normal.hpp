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
#include <cppmc/mcmc.hyperprior.hpp>

namespace CppMC {

  template<template<typename> class ArmaT>
  class Normal : public MCMCStochastic<double,ArmaT> {
  private:
    MCMCSpecialized<double,Col>& mu_;
    MCMCSpecialized<double,Col>& tau_;

  public:
    Normal(MCMCSpecialized<double,Col>& mu, MCMCSpecialized<double,Col>& tau_, const ArmaT<double> shape):
      MCMCStochastic<double,ArmaT>(shape),
      mu_(mu), tau_(tau_)
      {}

    // convenience wrapper for imlied hyperpriors
    Normal(const double mu, const double tau, const ArmaT<double> shape):
      MCMCStochastic<double,ArmaT>(shape),
      mu_(*(new HyperPrior<double,Col>(mu))), tau_(*(new HyperPrior<double,Col>(tau))) {
      // have to put the implicit hyperpriors on the locals list so they get deleted
      MCMCStochastic<double,ArmaT>::locals_.push_back(&mu_);
      MCMCStochastic<double,ArmaT>::locals_.push_back(&tau_);
    }

    double logp() const {
      double ans(0);
      const uint mu_size = mu_.size();
      const uint tau_size = tau_.size();
      for(size_t i = 0; i < MCMCStochastic<double,ArmaT>::size(); i++) {
	ans += normal_logp(MCMCStochastic<double,ArmaT>::value_[i], mu_[i % mu_size], tau_[i % tau_size]);
      }
      return ans;
    }

    void getParents(std::vector<MCMCObject*>& parents) const {
      parents.push_back(&mu_);
      parents.push_back(&tau_);
    }
  };
} // namespace CppMC
#endif // MCMC_NORMAL_HPP
