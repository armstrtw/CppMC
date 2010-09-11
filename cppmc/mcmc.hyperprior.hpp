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

  template<typename DataT,
           template<typename> class ArmaT>
  class HyperPrior : public MCMCSpecialized<DataT,ArmaT> {
  public:

    // init w/ existing armadillo object
    HyperPrior(const ArmaT<DataT>& shape) : MCMCSpecialized<DataT,ArmaT>(shape) {};

    // init w/ scalar (assumes vec is ArmaT, b/c mat can't be initialized w/ 'x(1)')
    HyperPrior(const DataT value) : MCMCSpecialized<DataT,ArmaT>(ArmaT<DataT>(1)) {
      MCMCSpecialized<DataT,ArmaT>::value_[0] = value;
    };
    void getParents(std::vector<MCMCObject*>& parents) const {}
    void jump() {}
    void update() {}
    void preserve() {}
    void revert() {}
    void tally() {}
    double logp() const { return static_cast<double>(0); }
    void print() const { cout << MCMCSpecialized<DataT,ArmaT>::value_ << endl; }
    bool isDeterministc() const { return false; }
    bool isStochastic() const { return false; }
  };
} // namespace CppMC
#endif // MCMC_HYPERPRIOR_HPP
