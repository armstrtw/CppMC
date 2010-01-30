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

  const double neg_inf(-std::numeric_limits<double>::infinity());

  template<typename T>
  class MCMCStochastic : public MCMCSpecialized<T> {
  protected:
    int iteration_;
    MCMCJumper<T> jumper_;
  public:
    MCMCStochastic(const T& shape): MCMCSpecialized<T>(shape), iteration_(-1), jumper_(MCMCSpecialized<T>::value_) {}
    virtual double logp() const = 0;
    void jump(const int current_iteration) {
      // only jump if we hevn't already jumped yet
      if(iteration_ == current_iteration) {
	return;
      }
      ++iteration_;
      jumper_.jump();
    }
    void revert() {
      jumper_.revert();
    }
    void tune(const double acceptance_rate) {
      jumper_.tune(acceptance_rate);
    }
  };
} // namespace CppMC
#endif // MCMC_STOCHASTIC_HPP
