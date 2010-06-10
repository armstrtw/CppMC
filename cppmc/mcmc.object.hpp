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

#include <vector>
#include <set>
#include <boost/random.hpp>

namespace CppMC {
  using namespace boost;

  // hardcoded for now
  typedef minstd_rand CppMCGeneratorT;
  //const double neg_inf(-std::numeric_limits<double>::infinity());

  class MCMCObject {
  protected:
    static CppMCGeneratorT generator_;
    std::vector<MCMCObject*> locals_;
  public:
    ~MCMCObject() { for(size_t i =0; i < locals_.size(); i++) { delete locals_[i]; } }
    MCMCObject() {}
    static void seedRNG(unsigned int s) { generator_.seed(s); }
    static void burnRNG(unsigned int n) { for(unsigned int i = 0; i < n; i++) { generator_(); } }
    static double tau_to_sd(const double tau) { return 1/sqrt(tau); }
    static double sd_to_tau(const double sd) { return 1/pow(sd,2.0); }

    // pure virtuals
    virtual void getParents(std::vector<MCMCObject*>& parents) const = 0; // user must provide this function to make object aware of parents
    virtual void jump() = 0;            // stocastics will jump values, determinsitics do nothing
    virtual void update() = 0;          // determinsitics jump values, stocastics do nothing
    virtual void preserve() = 0;        // in mcmc.specialized
    virtual void revert() = 0;          // in mcmc.specialized
    virtual void tally() = 0;           // in mcmc.specialized
    virtual double logp() const = 0;    // must be implemented for each specific distribution or likelihood
    virtual void print() const = 0;     // in mcmc.specialized

    std::vector<MCMCObject*> uniqueMCMCObjectList(std::vector<MCMCObject*>& x) {
      std::vector<MCMCObject*> ans;
      std::set<MCMCObject*> unqique_ans;
      for(size_t i = 0; i < x.size(); i++) {
        unqique_ans.insert(x[i]);
      }     
      std::copy(unqique_ans.begin(),unqique_ans.end(),std::back_inserter(ans));
      return ans;
    }

    void buildMCMCObjectList(std::vector<MCMCObject*>& x) {
      // push self
      x.push_back(this);

      std::vector<MCMCObject*> immediateParents;
      getParents(immediateParents);
      
      // put immedates on list
      std::copy(immediateParents.begin(),immediateParents.end(),std::back_inserter(x));
      
      //recursive to parents
      for(size_t i = 0; i < immediateParents.size(); i++) {
        immediateParents[i]->buildMCMCObjectList(x);
      }
    }
    void jump_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->jump(); } }
    void update_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->update(); } }
    void preserve_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->preserve(); } }
    void revert_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->revert(); } }
    void tally_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->tally(); } }
    double logp_all(std::vector<MCMCObject*>& v) { double ans(0); for(size_t i = 0; i < v.size(); i++) { ans += v[i]->logp(); } return ans; }
    void print_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->print(); } }
  };
} // namespace CppMC
#endif // MCMC_OBJECT_HPP
