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
#include <iostream>
#include <boost/random.hpp>

namespace CppMC {
  using namespace boost;

  // hardcoded for now
  typedef minstd_rand CppMCGeneratorT;

  class MCMCObject {
  protected:
    static CppMCGeneratorT generator_;
    boost::normal_distribution<double> normal_rng_dist_;
    boost::uniform_real<double> uniform_rng_dist_;
    boost::variate_generator<CppMCGeneratorT&, boost::normal_distribution<double> > rng_;
    boost::variate_generator<CppMCGeneratorT&, uniform_real<double> > uni_rng_;
    std::vector<MCMCObject*> locals_;
  public:
    ~MCMCObject() { for(size_t i =0; i < locals_.size(); i++) { delete locals_[i]; } }
    MCMCObject(): normal_rng_dist_(0, 1.0), uniform_rng_dist_(0.0, 1.0), rng_(generator_, normal_rng_dist_), uni_rng_(generator_, uniform_rng_dist_) {}
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
    virtual bool isDeterministc() const = 0;
    virtual bool isStochastic() const = 0;

    void uniqueMCMCObjectList(std::vector<MCMCObject*>& in, std::vector<MCMCObject*>& out) {
      std::set<MCMCObject*> unqique_values;
      for(size_t i = 0; i < in.size(); i++) {
        unqique_values.insert(in[i]);
      }
      std::copy(unqique_values.begin(),unqique_values.end(),std::back_inserter(out));
    }

    void buildMCMCObjectList(std::vector<MCMCObject*>& ans) {
      std::vector<MCMCObject*> aux_ans;
      buildMCMCObjectListAux(aux_ans);
      uniqueMCMCObjectList(aux_ans, ans);
    }

    void buildDeterministicList(std::vector<MCMCObject*>& in, std::vector<MCMCObject*>& out) {
      for(size_t i = 0; i < in.size(); i++)
        if(in[i]->isDeterministc())
          out.push_back(in[i]);
    }
    void buildStochasticList(std::vector<MCMCObject*>& in, std::vector<MCMCObject*>& out) {
      for(size_t i = 0; i < in.size(); i++)
        if(in[i]->isStochastic())
          out.push_back(in[i]);
    }
    void buildMCMCObjectListAux(std::vector<MCMCObject*>& ans) {
      // push self
      ans.push_back(this);

      std::vector<MCMCObject*> immediateParents;
      getParents(immediateParents);

      // put immedates on list
      std::copy(immediateParents.begin(),immediateParents.end(),std::back_inserter(ans));
      
      //recursive to parents
      for(size_t i = 0; i < immediateParents.size(); i++) {
        immediateParents[i]->buildMCMCObjectList(ans);
      }
    }
    void jump_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->jump(); } }
    void update_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->update(); } }
    void preserve_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->preserve(); } }
    void revert_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->revert(); } }
    void tally_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->tally(); } }
    double logp_all(std::vector<MCMCObject*>& v) { double ans(0); for(size_t i = 0; i < v.size(); i++) { ans += v[i]->logp(); } return ans; }
    // double logp_all(std::vector<MCMCObject*>& v) {
    //   double ans(0);
    //   for(size_t i = 0; i < v.size(); i++) {
    //     std::cout << "addr:" << v[i] << ":logp: " << v[i]->logp() << std::endl;
    //     ans += v[i]->logp();
    //   }
    //   return ans;
    // }
    void print_all(std::vector<MCMCObject*>& v) { for(size_t i = 0; i < v.size(); i++) { v[i]->print(); } }

    void sample(int iterations, int burn, int thin) {
      // for acceptace test
      boost::uniform_real<> uni_dist(0,1);
      boost::variate_generator<CppMCGeneratorT&, boost::uniform_real<> > uni_rng(generator_, uni_dist);

      double logp_value,old_logp_value;
      double accepted(0);
      double rejected(0);

      std::vector<MCMCObject*> mcmcObjects, stochastics, deterministics;
      buildMCMCObjectList(mcmcObjects);
      buildDeterministicList(mcmcObjects,deterministics);
      buildStochasticList(mcmcObjects,stochastics);
      std::cout << "mcmcObjects size: " << mcmcObjects.size() << std::endl;
      for(size_t i = 0; i < stochastics.size(); i++) {
        std::cout << "stoch addr:" << stochastics[i] << std::endl;
        stochastics[i]->print();
      }

      for(size_t i = 0; i < deterministics.size(); i++) {
        std::cout << "deterministic addr:" << deterministics[i] << std::endl;
        deterministics[i]->print();
      }

      logp_value  = -std::numeric_limits<double>::infinity();
      old_logp_value = -std::numeric_limits<double>::infinity();
      for(int i = 0; i < iterations; i++) {
        old_logp_value = logp_value;
        preserve_all(mcmcObjects);
        jump_all(stochastics); // only jump stocastics
        update_all(deterministics); // only update deterministics
	logp_value = logp_all(mcmcObjects);
        //std::cout << "logp: " << logp_value << std::endl;
	if(logp_value == -std::numeric_limits<double>::infinity() || log(uni_rng()) > logp_value - old_logp_value) {
	  revert_all(mcmcObjects);
          logp_value = old_logp_value;
	  rejected += 1;
	} else {
	  accepted += 1;
	}
	if(i > burn && i % thin == 0) {
          //std::cout << "ar: " << accepted / (accepted + rejected) << std::endl;
          accepted = 0;
          rejected = 0;
	  tally_all(mcmcObjects);
	}
      }
    }

  };
} // namespace CppMC
#endif // MCMC_OBJECT_HPP
