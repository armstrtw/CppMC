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
    static base_generator_type generator_;
    boost::normal_distribution<double> rng_dist_;
    boost::variate_generator<base_generator_type&, boost::normal_distribution<double> > rng_;
  public:
    MCMCJumperBase(): rng_dist_(0, 1.0), rng_(generator_, rng_dist_) {}
  };

  template<typename T>
  class MCMCJumper : public MCMCJumperBase {
  private:
    T& value_;
    T old_value_;
    T sd_;
    double scale_;
    void drawRNG() {
      for(size_t i = 0; i < nrow(value_) * ncol(value_); i++) {
	value_[i] += scale_ * sd_[i] * rng_();
      }
    }
  public:
    MCMCJumper(T& value): value_(value), old_value_(value), sd_(value), scale_(1.0) {
      sd_.fill(1.0);
    }
    void setSD(const double sd) {
      for(size_t i = 0; i < ncol(sd_) * nrow(sd_); i++) {
	sd_[i] = sd;
      }
    }
    void setScale(const double scale) {
      scale_ = scale;
    }
    void jump() {
      old_value_ = value_;
      drawRNG();
    }
    void revert() {
      value_ = old_value_;
    }

    void tune(const double acceptance_rate) {
      //cout << "acceptance_rate: " << acceptance_rate << endl;
      if(acceptance_rate < .01) {
	scale_ *= .1;
	return;
      }
      if(acceptance_rate < .05) {
	scale_ *= .5;
	return;
      }
      if(acceptance_rate < .2) {
	scale_ *= .9;
	return;
      }
      if(acceptance_rate < .5) {
	scale_ *= 1.1;
	return;
      }
      if(acceptance_rate > .95) {
	scale_ *= 10;
	return;
      }
      if(acceptance_rate > .75) {
	scale_ *= 2;
	return;
      }
    }
  };
} // namespace
#endif // MCMC_JUMPER_HPP
