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
#include <armadillo>
#include <cppmc/mcmc.object.hpp>

namespace CppMC {
  using namespace arma;

  template<typename DataT,
           template<typename> class ArmaT>
  class MCMCSpecialized : public MCMCObject {
  protected:
    ArmaT<DataT> value_;
    ArmaT<DataT> old_value_;
    std::vector< ArmaT<DataT> > history_;
  public:
    MCMCSpecialized(const ArmaT<DataT>& shape): MCMCObject(), value_(shape) {}
    const ArmaT<DataT>& exposeValue() const {
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
    const std::vector< ArmaT<DataT> >& getHistory() const {
      return history_;
    }

    uint nrow() const { return value_.n_rows; }
    uint ncol() const { return value_.n_cols; }
    uint size() const { return nrow() * ncol(); }

    void shape(std::vector<uint>& ans) const {
      ans.push_back(nrow());
      ans.push_back(ncol());
    }

    std::vector<uint> shape() const {
      std::vector<int> ans(2);
      ans[0] = nrow();
      ans[1] = ncol();
      return ans;
    }

    // allow user to subscript this object directly
    DataT& operator[](const int i) {
      return value_(i);
    }
    DataT& operator()(const int i, const int j) {
      return value_(i,j);
    }
    void fill(const DataT fill_value) {
      value_.fill(fill_value);
    }
  };
} // namespace CppMC
#endif // MCMC_SPECIALIZED_HPP
