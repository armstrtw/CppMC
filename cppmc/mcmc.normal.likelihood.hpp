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

#include <cppmc/mcmc.object.hpp>
#include <cppmc/mcmc.logp.functions.hpp>

namespace CppMC {

  template<template<typename> class ArmaT>
  class NormalLikelihood : public MCMCObject {
  private:
    const ArmaT<double>& observations_;
    MCMCSpecialized<double,ArmaT>& forecast_;
    MCMCSpecialized<double,ArmaT>& tau_;
  public:
    NormalLikelihood(const ArmaT<double>& observations, MCMCSpecialized<double,ArmaT>& forecast, MCMCSpecialized<double,ArmaT>& tau): observations_(observations), forecast_(forecast), tau_(tau) {}
    double logp() const {
      double ans(0);
      const ArmaT<double>& sample = forecast_();
      const uint sample_size = sample.n_elem;
      const uint tau_size = tau_.size();
      for(uint i = 0; i < observations_.n_elem; i++) {
        ans += normal_logp(sample[i % sample_size], observations_[i], tau_[i % tau_size]);
      }
      return ans;
    }
    void getParents(std::vector<MCMCObject*>& parents) const {
      parents.push_back(&forecast_);
      parents.push_back(&tau_);
    }
    void jump() {}
    void update() {}
    void preserve() {}
    void revert() {}
    void tally() {}
    void print() const {
      cout << "actual" << endl;
      cout << observations_ << endl;
      cout << "forecast" << endl;
      cout << forecast_() << endl;
      cout << "tau" << endl;
      cout << tau_() << endl;
    }
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return false; }
  };
}
#endif // MCMC_NORMAL_LIKELIHOOD_HPP
