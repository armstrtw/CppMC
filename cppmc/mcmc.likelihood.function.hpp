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

#ifndef MCMC_LIKELIHOOD_FUNCTION_HPP
#define MCMC_LIKELIHOOD_FUNCTION_HPP

#include <iostream>
#include <cppmc/mcmc.object.hpp>
#include <cppmc/mcmc.specialized.hpp>

namespace CppMC {

  template<typename DataT,
           template<typename> class ArmaT>
  class LikelihoodFunction : public MCMCObject {
  protected:
    const ArmaT<DataT>& actual_values_;
    MCMCSpecialized<DataT,ArmaT>& forecast_;
  public:
    LikelihoodFunction(const ArmaT<DataT>& actual_values, MCMCSpecialized<DataT,ArmaT>& forecast): MCMCObject(), generator_(20u), uni_dist_(0,1), uni_rng_(generator_, uni_dist_), actual_values_(actual_values), forecast_(forecast) {}


    void getParents(std::vector<MCMCObject*>& parents) const {
      parents.push_back(&forecast_);
    }

    // don't need any of these
    void jump() {}
    void update() {}
    void preserve() {}
    void revert() {}
    void tally() {}
    void print() {
      std::cout << "acutal values:" << std::endl << actual_values_;
      std::cout << "forecast:" << std::endl << forecast_();
    }
    bool isDeterministc() const { return true; }
    bool isStochastic() const { return false; }
  };
} // namespace CppMC
#endif // MCMC_LIKELIHOOD_FUNCTION_HPP
