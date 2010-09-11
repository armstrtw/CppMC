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

#ifndef MCMC_STOCHASTIC_HPP
#define MCMC_STOCHASTIC_HPP

#include <cppmc/mcmc.specialized.hpp>

namespace CppMC {

  template<typename DataT,
           template<typename> class ArmaT>
  class MCMCStochastic : public MCMCSpecialized<DataT,ArmaT> {
  protected:
    double scale_;
  public:
    MCMCStochastic(const ArmaT<DataT>& shape): MCMCSpecialized<DataT,ArmaT>(shape), scale_(1.0) {}
    void jump() {
      for(size_t i = 0; i < MCMCSpecialized<DataT,ArmaT>::value_.n_elem; i++) {
	MCMCSpecialized<DataT,ArmaT>::value_[i] += scale_ * MCMCSpecialized<DataT,ArmaT>::rng_();
      }
    }
    void update() {}
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return true; }
  };
} // namespace CppMC
#endif // MCMC_STOCHASTIC_HPP
