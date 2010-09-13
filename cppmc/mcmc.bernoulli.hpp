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

#ifndef MCMC_BERNOULLI_HPP
#define MCMC_BERNOULLI_HPP

#include <iostream>
#include <cppmc/mcmc.stochastic.hpp>
#include <cppmc/mcmc.hyperprior.hpp>

namespace CppMC {


  template<template<typename> class ArmaT>
  class Bernoulli : public MCMCStochastic<uint,ArmaT> {
  private:
    MCMCSpecialized<double,Col>& p_;
  public:
    Bernoulli(MCMCSpecialized<double,Col>& p, const ArmaT<uint> shape):
      MCMCStochastic<uint,ArmaT>(shape), p_(p)
    {}

    // convenience wrapper for imlied hyperpriors
    Bernoulli(const double p, const ArmaT<double> shape):
      MCMCStochastic<uint,ArmaT>(shape),
      p_(*(new HyperPrior<double,Col>(p)))
    {
      // have to put the implicit hyperpriors on the locals list so they get deleted
      MCMCStochastic<uint,ArmaT>::locals_.push_back(&p_);
    }

    double logp() const {
      double ans(0);
      const size_t p_size = p_.size();
      for(size_t i = 0; i < MCMCStochastic<uint,ArmaT>::size(); i++) {
	ans += bernoulli_logp(MCMCStochastic<uint,ArmaT>::value_[i], p_[i % p_size]);
      }
      return ans;
    }

    void getParents(std::vector<MCMCObject*>& parents) const {
      parents.push_back(&p_);
    }

    //specialized jump
    void jump() {
      const size_t p_size = p_.size();
      for(size_t i = 0; i < MCMCStochastic<uint,ArmaT>::size(); i++) {
        MCMCStochastic<uint,ArmaT>::value_[i] = MCMCStochastic<uint,ArmaT>::uni_rng_() > p_[i % p_size] ? 1 : 0;
      }
    }
  };
} // namespace CppMC
#endif // MCMC_BERNOULLI_HPP
