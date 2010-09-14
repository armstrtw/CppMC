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
#include <iostream>
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
    //const ArmaT<DataT>& exposeValue() const { return value_; }
    const ArmaT<DataT>& operator()() const { return value_; }
    void preserve() { old_value_ = value_; }
    void revert() { value_ = old_value_; }
    void tally() { history_.push_back(value_); }
    void print() const { std::cout << value_ << std::endl; }
    const std::vector< ArmaT<DataT> >& getHistory() const { return history_; }
    uint nrow() const { return value_.n_rows; }
    uint ncol() const { return value_.n_cols; }
    uint size() const { return nrow() * ncol(); }

    void shape(std::vector<uint>& ans) const {
      if(nrow() > 0) { ans.push_back(nrow()); }
      if(ncol() > 0) { ans.push_back(ncol()); }
    }

    std::vector<uint> shape() const {
      std::vector<uint> ans;
      shape(ans);
      return ans;
    }

    // allow user to subscript this object directly
    DataT& operator[](const int i) {
      return value_[i];
    }
    DataT& operator()(const int i, const int j) {
      return value_(i,j);
    }
    void fill(const DataT fill_value) {
      value_.fill(fill_value);
    }

    ArmaT<double> mean() const {
      //FIXME: this may not work if value_ is not <double>
      // have to think of some thing better, possibly partial specialization
      // for Col<> and Mat<>
      //ArmaT<double> ans = conv_to<mat>::from(value_);
      ArmaT<double> ans;
      ans.copy_size(value_);
      ans.fill(0.0);
      for(size_t i = 0; i < history_.size(); i++) {
        ans += conv_to<mat>::from(history_[i]);
      }
      ans /= static_cast<double>(history_.size());
      return ans;
    }
  };
} // namespace CppMC
#endif // MCMC_SPECIALIZED_HPP
