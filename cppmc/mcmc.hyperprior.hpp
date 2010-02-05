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

#ifndef MCMC_HYPERPRIOR_HPP
#define MCMC_HYPERPRIOR_HPP

#include <cppmc/mcmc.specialized.hpp>

namespace CppMC {

  template<typename T>
  class HyperPrior : public MCMCSpecialized<DataT> {
  public:
    HyperPrior(const T& value) : MCMCSpecialized<DataT>(), MCMCSpecialized<DataT>::value_(value) {};
    double logp() const { return static_cast<double>(0); }
    void tally() {}
    void tally_parents() {}
  };
} // namespace CppMC
#endif // MCMC_HYPERPRIOR_HPP
