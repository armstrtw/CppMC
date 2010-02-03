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

#ifndef MCMC_LIKELIHOOD_FUNCTION_HPP
#define MCMC_LIKELIHOOD_FUNCTION_HPP

#include <cppmc/mcmc.object.hpp>
#include <cppmc/mcmc.specialized.hpp>

namespace CppMC {

  template<typename T>
  class LikelihoodFunction : public MCMCObject {
  private:
    // for acceptace test
    base_generator_type generator_;
    boost::uniform_real<> uni_dist_;
    boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni_rng_;
  protected:
    const T& actual_values_;
    MCMCSpecialized<T>& forecast_;
  public:
    LikelihoodFunction(const T& actual_values, MCMCSpecialized<T>& forecast): MCMCObject(), generator_(20u), uni_dist_(0,1), uni_rng_(generator_, uni_dist_), actual_values_(actual_values), forecast_(forecast) {
      registerParents();
    }

    void sample(int iterations, int burn, int thin) {
      double accepted(0);
      double rejected(0);

      for(int i = 0; i < iterations; i++) {
	double logp_old = logp();
	jump(i);
	double logp_new = logp();
	if(logp_new == neg_inf || log(uni_rng_()) > logp_new - logp_old) {
	  revert();
	  rejected+=1;
	} else {
	  accepted+=1;
	}

	// tune every 50 during burn
	if(i % 50 == 0 && i < burn) {
	  tune(accepted/(accepted + rejected));
	  accepted = 0;
	  rejected = 0;
	}

	// tune every 1000 during actual
	if(i % 1000 == 0) {
	  tune(accepted/(accepted + rejected));
	}
	if(i > burn && i % thin == 0) {
	  tally();
	}
      }
    }

    void registerParents() {
      MCMCObject::parents_.push_back(&forecast_);
    }
    
    // don't need any of these
    void jump_self() {}
    void preserve_self() {}
    void revert_self() {}
    void tally_self() {}
    void tune_self(const double acceptance_rate) {}
  };
} // namespace CppMC
#endif // MCMC_LIKELIHOOD_FUNCTION_HPP
