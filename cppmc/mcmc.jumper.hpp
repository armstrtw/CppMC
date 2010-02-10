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

#ifndef MCMC_JUMPER_HPP
#define MCMC_JUMPER_HPP

namespace CppMC {
  class MCMCJumperBase {
  protected:
    base_generator_type& generator_;
    boost::normal_distribution<double> rng_dist_;
    boost::variate_generator<base_generator_type&, boost::normal_distribution<double> > rng_;
  public:
    MCMCJumperBase(base_generator_type& generator): generator_(generator), rng_dist_(0, 1.0), rng_(generator_, rng_dist_) {}
  };

  template<typename DataT,
           template<typename> class ArmaT>
  class MCMCJumper : public MCMCJumperBase {
  private:
    ArmaT<DataT>& value_;
    ArmaT<DataT> sd_;
    double scale_;
  public:
    MCMCJumper(ArmaT<DataT>& value, base_generator_type& generator):
      MCMCJumperBase(generator), value_(value), sd_(value), scale_(1.0) {
      sd_.fill(1.0);
    }
    void setSD(const ArmaT<DataT> sd) { sd_ = sd; }
    void setScale(const double scale) { scale_ = scale; }
    void jump() {
      for(size_t i = 0; i < value_.n_elem; i++) {
	value_[i] += scale_ * sd_[i] * rng_();
      }
    }
    void tune(const double acceptance_rate) {
      //cout << "acceptance_rate: " << acceptance_rate << endl;
      if(acceptance_rate < .001) {
	scale_ *= .1;
      } else if(acceptance_rate < .05) {
	scale_ *= .5;
      } else if(acceptance_rate < .2) {
	scale_ *= .9;
      } else if(acceptance_rate > .95) {
	scale_ *= 10;
      } else if(acceptance_rate > .75) {
	scale_ *= 2;
      } else if(acceptance_rate > .5) {
	scale_ *= 1.1;
      }
    }
  };
} // namespace
#endif // MCMC_JUMPER_HPP
