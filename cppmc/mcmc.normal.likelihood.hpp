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

#ifndef MCMC_NORMAL_LIKELIHOOD_HPP
#define MCMC_NORMAL_LIKELIHOOD_HPP

#include <cppmc/mcmc.likelihood.function.hpp>
#include <cppmc/mcmc.logp.functions.hpp>

namespace CppMC {

  template<template<typename> class ArmaT>
  class NormalLikelihood : public LikelihoodFunction<double,ArmaT> {
  private:
    const double tau_;
  public:
    NormalLikelihood(const ArmaT<double>& actual_values, MCMCSpecialized<double,ArmaT>& forecast, const double standard_deviation): LikelihoodFunction<double,ArmaT>(actual_values, forecast), tau_(MCMCObject::sd_to_tau(standard_deviation)) {}

    double calc_logp_self() const {
      double ans(0);    
      const ArmaT<double>& sample = LikelihoodFunction<double,ArmaT>::forecast_.exposeValue();
      for(uint i = 0; i < LikelihoodFunction<double,ArmaT>::actual_values_.n_elem; i++) {
        ans += normal_logp(sample[i], LikelihoodFunction<double,ArmaT>::actual_values_[i], tau_);
      }
      return ans;
    }
  };
}
#endif // MCMC_NORMAL_LIKELIHOOD_HPP
