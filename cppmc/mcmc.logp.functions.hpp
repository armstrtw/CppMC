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

#ifndef MCMC_LOGP_FUNCTIONS_HPP
#define MCMC_LOGP_FUNCTIONS_HPP

#include <cmath>
#include <armadillo>

namespace CppMC {
  using std::log;

  inline double discrete_uniform_logp(const double x, const double lower, const double upper) {
      return (x < lower || x > upper) ? -std::numeric_limits<double>::infinity() : -log(upper - lower + 1.0);
  }

  inline double uniform_logp(const double x, const double lower, const double upper) {
      return (x < lower || x > upper) ? -std::numeric_limits<double>::infinity() : -log(upper - lower);
  }

  inline double normal_logp(const double x, const double mu, const double tau) {
    return 0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(x-mu,2);
  }

  inline double truncated_normal_logp(const double x, const double mu, const double tau, const double a, const double b) {
    // FIXME: this is incorrect, but just get it working in stubs for now    
    return (x < a || x > b) ? -std::numeric_limits<double>::infinity() : 0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(x-mu,2);
  }

  inline double bernoulli_logp(const int x, const double p) {
    return x ? log(p) : log(1 - p);
  }

} // namespace CppMC
#endif // MCMC_LOGP_FUNCTIONS_HPP
