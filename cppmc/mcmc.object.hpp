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

#ifndef MCMC_OBJECT_HPP
#define MCMC_OBJECT_HPP

// should be settable by the user somewhere...
typedef boost::minstd_rand base_generator_type;

namespace CppMC {
  class MCMCObject {
  protected:
    static base_generator_type generator_;
  public:
    MCMCObject() {}
    virtual double logp() const = 0;
    virtual void jump(int current_iteration) = 0;
    virtual void revert() = 0;
    virtual void tally() = 0;
    virtual void tally_parents() = 0;
    virtual void tune(const double acceptance_rate) = 0;
  };
} // namespace CppMC
#endif // MCMC_OBJECT_HPP
