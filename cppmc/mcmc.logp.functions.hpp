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

  template<typename T>
  inline
  double discrete_uniform_logp(const T x, const T lower, const T upper) {
      return (x < lower || x > upper) ? neg_inf : -log(upper - lower + 1.0);
  }

  template<typename T>
  inline
  double uniform_logp(const T x, const T lower, const T upper) {
      return (x < lower || x > upper) ? neg_inf : -log(upper - lower);
  }
  template<typename T>
  inline
  double normal_logp(const T x, const T mu, const T tau) {
    return 0.5*log(0.5*tau/arma::math::pi()) - 0.5 * tau * pow(x-mu,2);
  }
  /*
  template<typename T>
  double exponentiated_logp(const T x, const T lower, const T upper) {
    t1 = dexp(-z(i)**cc);
    pdf = aa*cc*(1.0-t1)**(aa-1.0)*t1*z(i)**(cc-1.0);
    return log(pdf/sigma);
  }

  template<typename T>
  double poisson_logp(const T x, const T mu) {
    if(x < 0 || mu < 0) {
      return neg_inf;
    }
    return x * log(mu) - mu - factln(x);
  }
  //SUBROUTINE trpoisson(x,mu,k,n,nmu,like)
  template<typename T>
  double truncated_poisson_logp(const T x, const T mu, const T k) {
    if(x < 0 || mu < 0 || k < 0) {
      return neg_inf;
    }
    double sumx = x * log(mu) - mu;
    double sumfact = factln(x);
    double cdf = gammq(dble(k), mu);
    double sumcdf =  log(1 -cdf);
    return sumx - sumfact - sumcdf;
  }

  //SUBROUTINE t(x,nu,n,nnu,like)
  template<typename T>
  double student_t_logp(const T x, const T nu) {
    if(nu <= 0) {
      return neg_inf;
    }
    return gammln((nu+1.0)/2.0) - 0.5*log(nu * arma::math::pi()) - gammln(nu/2.0) - (nu+1)/2 * dog(1 + pow(x,2)/nu);
  }

  //SUBROUTINE multinomial(x,n,p,nx,nn,np,k,like)
  template<typename T>
  double multinomial_logp(x,n,p,k) {
      return neg_inf;
  }
  */
} // namespace CppMC
#endif // MCMC_LOGP_FUNCTIONS_HPP
