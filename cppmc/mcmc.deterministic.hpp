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

#ifndef MCMC_SPECIALIZED_OBJECT_HPP
#define MCMC_SPECIALIZED_OBJECT_HPP

#include <cppmc/mcmc.specialized.hpp>

namespace CppMC {

  template<typename T>
  class MCMCDeterministic : public MCMCSpecialized<T> {
  public:
    MCMCDeterministic(const Mat<T>& initial_value): MCMCSpecialized<T>(initial_value) {}

    // deterministics only derive their logp from their parents
    double calc_logp_self() const { return 0; }

    // assumes parents have already been updated
    void jump_self() { MCMCSpecialized<T>::value_ = eval(); }

    // no need to tune deterministic
    void tune_self(const double acceptance_rate) {}

    // user must provide this function to update object
    virtual Mat<T> eval() const = 0;
  };
} // namespace CppMC
#endif // MCMC_SPECIALIZED_OBJECT_HPP
