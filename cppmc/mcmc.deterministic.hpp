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

#ifndef MCMC_DETERMINISTIC_HPP
#define MCMC_DETERMINISTIC_HPP

#include <cppmc/mcmc.specialized.hpp>

namespace CppMC {

  template<typename DataT,
           template<typename> class ArmaT>
  class MCMCDeterministic : public MCMCSpecialized<DataT,ArmaT> {
  public:
    MCMCDeterministic(const ArmaT<DataT>& initial_value): MCMCSpecialized<DataT,ArmaT>(initial_value) {}

    // deterministics only derive their logp from their parents
    double logp() const { return 0; }

    // do nothing, object must be updated after all other objects are jumped
    void jump() {}
    void update() { MCMCSpecialized<DataT,ArmaT>::value_ = eval(); }
    bool isDeterministc() const { return true; }
    bool isStochastic() const { return false; }

    // user must provide this function to update object
    virtual ArmaT<DataT> eval() const = 0;
  };
} // namespace CppMC
#endif // MCMC_DETERMINISTIC_HPP
