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
#include <cppmc/mcmc.jumper.hpp>

namespace CppMC {

  template<typename DataT,
           template<typename> class ArmaT>
  class MCMCStochastic : public MCMCSpecialized<DataT,ArmaT> {
  protected:
    MCMCJumper<DataT,ArmaT> jumper_;
  public:
    MCMCStochastic(const ArmaT<DataT>& shape):
      MCMCSpecialized<DataT,ArmaT>(shape),
      jumper_(MCMCSpecialized<DataT,ArmaT>::value_,MCMCSpecialized<DataT,ArmaT>::generator_) {}
    void jump_self() {
      jumper_.jump();
    }
  };
} // namespace CppMC
#endif // MCMC_STOCHASTIC_HPP
