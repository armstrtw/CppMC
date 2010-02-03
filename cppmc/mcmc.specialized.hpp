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

#ifndef MCMC_SPECIALIZED_HPP
#define MCMC_SPECIALIZED_HPP

#include <vector>
#include <cppmc/mcmc.object.hpp>

namespace CppMC {

  template<typename T>
  class MCMCSpecialized : public MCMCObject {
  protected:
    T value_;
    T old_value_;
    std::vector<T> history_;
  public:
    MCMCSpecialized(const T& shape): MCMCObject(), value_(shape) {}
    const T& exposeValue() const {
      return value_;
    }
    void preserve_self() {
      old_value_ = value_;
    }
    void revert_self() {
      value_ = old_value_;
    }
    void tally_self() {
      history_.push_back(value_);
    }
    const std::vector<T>& getHistory() const {
      return history_;
    }
  };
} // namespace CppMC
#endif // MCMC_SPECIALIZED_HPP
