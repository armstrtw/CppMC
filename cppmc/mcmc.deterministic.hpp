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
  protected:
    std::vector<MCMCObject*> parents_;
  public:
    MCMCDeterministic(const T& initial_value): MCMCSpecialized<T>(initial_value) {}
    double logp() const {
      double ans(0);
      for(std::vector<MCMCObject*>::const_iterator  iter = parents_.begin(); iter!=parents_.end(); iter++) {
	ans += (*iter)->logp();
      }
      return ans;
    }
    void jump(int current_iteration) {
      for(std::vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
	(*iter)->jump(current_iteration);
      }
      MCMCSpecialized<T>::value_ = eval();
    }
    void revert() {
      for(std::vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
	(*iter)->revert();
      }
      MCMCSpecialized<T>::value_ = eval();
    }
    void tally_parents() {
      for(std::vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
	(*iter)->tally();
      }
    }
    void tune(const double acceptance_rate) {
      for(std::vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
	(*iter)->tune(acceptance_rate);
      }
    }
    virtual void registerParents() = 0; // user must provide this function to make object aware of parents
    virtual T eval() = 0;  // user must provide this function to update object
  };
} // namespace CppMC
#endif // MCMC_SPECIALIZED_OBJECT_HPP
