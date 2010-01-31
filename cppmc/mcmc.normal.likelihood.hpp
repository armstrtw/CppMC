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

#include <armadillo>
#include <cppmc/mcmc.likelihood.function.hpp>
#include <cppmc/mcmc.logp.functions.hpp>

namespace CppMC {

template<typename T>
class NormalLikelihood : public LikelihoodFunction<T> {
 private:
  const double tau_;
public:
  NormalLikelihood(const T& actual_values, MCMCDeterministic<T>& forecaster, const double tau): LikelihoodFunction<T>(actual_values, forecaster), tau_(tau) {}

  double logp() const {
    double ans(0);
    const T& sample = LikelihoodFunction<T>::forecaster_.exposeValue();
    for(int i = 0; i < LikelihoodFunction<T>::actual_values_.n_elem; i++) {
      ans += normal_logp(sample[i], LikelihoodFunction<T>::actual_values_[i], tau_);
    }
    ans += LikelihoodFunction<T>::forecaster_.logp();
    return ans;
  }
};
}
#endif // MCMC_NORMAL_LIKELIHOOD_HPP
