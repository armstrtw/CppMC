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

#ifndef MCMC_TRUNCATED_NORMAL_HPP
#define MCMC_TRUNCATED_NORMAL_HPP

#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>
#include <cppmc/mcmc.stochastic.hpp>
#include <cppmc/mcmc.hyperprior.hpp>

namespace CppMC {

  template<template<typename> class ArmaT>
  class TruncatedNormal : public MCMCStochastic<double,ArmaT> {
  private:
    MCMCSpecialized<double,Col>& mu_;
    MCMCSpecialized<double,Col>& tau_;
    MCMCSpecialized<double,Col>& a_;
    MCMCSpecialized<double,Col>& b_;
  public:
    TruncatedNormal(MCMCSpecialized<double,Col>& mu, MCMCSpecialized<double,Col>& tau, MCMCSpecialized<double,Col>& a, MCMCSpecialized<double,Col>& b, const ArmaT<double> shape):
      MCMCStochastic<double,ArmaT>(shape),
      mu_(mu), tau_(tau), a_(a), b_b(b)
      {}

    // convenience wrapper for imlied hyperpriors
    TruncatedNormal(const double mu, const double tau, const double a, const double b, const ArmaT<double> shape):
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
      const uint a_size = a_.size();
      const uint b_size = b_.size();
      for(size_t i = 0; i < MCMCStochastic<double,ArmaT>::size(); i++) {
	ans += truncated_normal_logp(MCMCStochastic<double,ArmaT>::value_[i], mu_[i % mu_size], tau_[i % tau_size], a_[i % a_size], b_[i % b_size]);
      }
      return ans;
    }

    void getParents(std::vector<MCMCObject*>& parents) const {
      parents.push_back(&mu_);
      parents.push_back(&tau_);
      parents.push_back(&a_);
      parents.push_back(&b_);
    }
  };
} // namespace CppMC
#endif // MCMC_TRUNCATED_NORMAL_HPP
