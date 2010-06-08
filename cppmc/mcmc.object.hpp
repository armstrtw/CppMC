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

#include <boost/random.hpp>

namespace CppMC {
  using namespace boost;

  // hardcoded for now
  typedef minstd_rand CppMCGeneratorT;
  const double neg_inf(-std::numeric_limits<double>::infinity());

  class MCMCObject {
  protected:
    static CppMCGeneratorT generator_;
    int iteration_;
    double logp_;
    double old_logp_;
    std::vector<MCMCObject*> parents_;
    std::vector<MCMCObject*> locals_;
  public:
    ~MCMCObject() { for(size_t i =0; i < locals_.size(); i++) { delete locals_[i]; } }
    MCMCObject(): iteration_(-1), logp_(neg_inf), old_logp_(neg_inf) {}
    static void seedRNG(unsigned int s) { generator_.seed(s); }
    static void burnRNG(unsigned int n) { for(unsigned int i = 0; i < n; i++) { generator_(); } }
    double logp() const { return logp_self() + logp_parents(); }
    double logp_self() const { return logp_; }
    static double tau_to_sd(const double tau) { return 1/sqrt(tau); }
    static double sd_to_tau(const double sd) { return 1/pow(sd,2.0); }
    void jump(const int current_iteration) {
      // only jump if we hevn't already jumped yet
      if(iteration_ == current_iteration) {
	return;
      }
      ++iteration_;

      old_logp_ = logp_;               // preserve logp value
      jump_parents(current_iteration); // jump parents before jump self
      preserve_self();
      jump_self();                     // node specific jump
      logp_ = calc_logp_self();        // update logp value
    }

    void revert() {
      logp_ = old_logp_;
      revert_self();       // in mcmc.specialized
      revert_parents();
    }

    void tally() {
      tally_self();        // in mcmc.specialized
      tally_parents();
    }

    void print() {
      print_self();

      // print parents
      for(std::vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
	(*iter)->print();
      }
    }


    // idea is to provide all the parent methods as virtuals
    // so that child classes can override and not use the vector iterators
    // b/c each child will already know who its parents are
    // hence no need for loop overhead

    virtual double logp_parents() const {
      double ans(0);
      for(std::vector<MCMCObject*>::const_iterator  iter = parents_.begin(); iter!=parents_.end(); iter++) {
	ans += (*iter)->logp();
      }
      return ans;
    }

    virtual void revert_parents() {
      for(std::vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
	(*iter)->revert();
      }
    }

    virtual void jump_parents(const int current_iteration) {
      for(std::vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
	(*iter)->jump(current_iteration);
      }
    }

    virtual void tally_parents() {
      for(std::vector<MCMCObject*>::iterator iter = parents_.begin(); iter!=parents_.end(); iter++) {
	(*iter)->tally();
      }
    }

    // pure virtuals
    virtual void registerParents() = 0; // user must provide this function to make object aware of parents
    virtual double calc_logp_self() const = 0;  // must be implemented for each specific distribution or likelihood
    virtual void jump_self() = 0; // stocastics will jump values, determinsitics will jump parents & update
    virtual void preserve_self() = 0;       // in mcmc.specialized
    virtual void revert_self() = 0;       // in mcmc.specialized
    virtual void tally_self() = 0;        // in mcmc.specialized
    virtual void print_self() = 0;        // in mcmc.specialized
  };
} // namespace CppMC
#endif // MCMC_OBJECT_HPP
