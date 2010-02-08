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

    boost::normal_distribution<double> rng_dist_;
    boost::variate_generator<base_generator_type&, boost::normal_distribution<double> > rng_;
  public:
    Normal(MCMCSpecialized<double,Col>& mu, MCMCSpecialized<double,Col>& tau_, const ArmaT<double> shape):
      MCMCStochastic<double,ArmaT>(shape),
      mu_(mu), tau_(tau_),
      rng_dist_(0, 1), rng_(MCMCStochastic<double,ArmaT>::generator_, rng_dist_) {

      // set values
      for(size_t i = 0; i < MCMCStochastic<double,ArmaT>::size(); i++) {
        // should throw exception if sizes are incompatible (or odd)
        // if mu is scalar, then wrapping is behavior we want
        const uint mu_size = mu_.size();
        for(uint row = 0; row < MCMCStochastic<double,ArmaT>::value_.n_rows; row++) {
          for(uint col = 0; col < MCMCStochastic<double,ArmaT>::value_.n_cols; col++) {
            MCMCStochastic<double,ArmaT>::value_(row,col) = rng_() + mu_[col % mu_size];
          }
        }
      }
    }

    // convenience wrapper for imlied hyperpriors
    Normal(const double mu, const double tau, const ArmaT<double> shape):
      MCMCStochastic<double,ArmaT>(shape),
      mu_(*(new HyperPrior<double,Col>(mu))), tau_(*(new HyperPrior<double,Col>(tau))),
      rng_dist_(0, 1), rng_(MCMCStochastic<double,ArmaT>::generator_, rng_dist_) {

      // have to put the implicit hyperpriors on the locals list so they get deleted
      MCMCStochastic<double,ArmaT>::locals_.push_back(&mu_);
      MCMCStochastic<double,ArmaT>::locals_.push_back(&tau_);

      // set values
      for(size_t i = 0; i < MCMCStochastic<double,ArmaT>::size(); i++) {
        // should throw exception if sizes are incompatible (or odd)
        // if mu is scalar, then wrapping is behavior we want
        const uint mu_size = mu_.size();
        for(uint row = 0; row < MCMCStochastic<double,ArmaT>::value_.n_rows; row++) {
          for(uint col = 0; col < MCMCStochastic<double,ArmaT>::value_.n_cols; col++) {
            MCMCStochastic<double,ArmaT>::value_(row,col) = rng_() + mu_[col % mu_size];
          }
        }
      }
    }

    double calc_logp_self() const {
      double ans(0);
      for(size_t i = 0; i < MCMCStochastic<double,ArmaT>::size(); i++) {
        const uint mu_size = mu_.size();
        const uint tau_size = tau_.size();
	ans += normal_logp(MCMCStochastic<double,ArmaT>::value_[i], mu_[i % mu_size], tau_[i % tau_size]);
      }
      return ans;
    }

    // need to define when mu and sd are allowed to be MCMC objects
    void registerParents() {
      MCMCStochastic<double,ArmaT>::parents_.push_back(&mu_);
      MCMCStochastic<double,ArmaT>::parents_.push_back(&tau_);
    }
  };
} // namespace CppMC
#endif // MCMC_NORMAL_HPP
