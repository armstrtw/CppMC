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
#include <cppmc/mcmc.hyperprior.hpp>

namespace CppMC {


  template<template<typename> class ArmaT>
  class Uniform : public MCMCStochastic<double,ArmaT> {
  private:
    MCMCSpecialized<double,Col>& lower_bound_;
    MCMCSpecialized<double,Col>& upper_bound_;
  public:
    Uniform(MCMCSpecialized<double,Col>& lower_bound, MCMCSpecialized<double,Col>&upper_bound, const ArmaT<double> shape):
      MCMCStochastic<double,ArmaT>(shape),
      lower_bound_(lower_bound), upper_bound_(upper_bound)
    {}

    // convenience wrapper for imlied hyperpriors
    Uniform(const double lower_bound, const double upper_bound, const ArmaT<double> shape):
      MCMCStochastic<double,ArmaT>(shape),
      lower_bound_(*(new HyperPrior<double,Col>(lower_bound))), upper_bound_(*(new HyperPrior<double,Col>(upper_bound)))
    {
      // have to put the implicit hyperpriors on the locals list so they get deleted
      MCMCStochastic<double,ArmaT>::locals_.push_back(&lower_bound_);
      MCMCStochastic<double,ArmaT>::locals_.push_back(&upper_bound_);
    }

    double logp() const {
      double ans(0);
      const uint lower_size = lower_bound_.size();
      const uint upper_size = upper_bound_.size();
      for(size_t i = 0; i < MCMCStochastic<double,ArmaT>::size(); i++) {
	ans += uniform_logp(MCMCStochastic<double,ArmaT>::value_[i], lower_bound_[i % lower_size], upper_bound_[i % upper_size]);
      }
      return ans;
    }

    void getParents(std::vector<MCMCObject*>& parents) const {
      parents.push_back(&lower_bound_);
      parents.push_back(&upper_bound_);
    }

    // specialized jump for uniform
    void jump() { 
      MCMCStochastic<double,ArmaT>::jumper_.jump();

      // neg inf jump, redo immediately
      while(logp() == -std::numeric_limits<double>::infinity()) {
        MCMCStochastic<double,ArmaT>::revert();
        MCMCStochastic<double,ArmaT>::jumper_.jump();
      }
    }
  };
} // namespace CppMC
#endif // MCMC_UNIFORM_HPP
