/*!
  \file model_polarized_movie_shearing_spot.cpp
  \author Paul Tiede
  \date  March, 2018
  \brief Implements extended shearing spot movie class.
*/

#include "model_polarized_movie_shearing_spot.h"
#include <cmath>

namespace Themis {

model_polarized_movie_shearing_spot::model_polarized_movie_shearing_spot(double start_observation, std::vector<double> observation_times, bool bkgd_riaf, double frequency, double M, double D)
  :model_polarized_movie<model_polarized_image_shearing_spot>(observation_times)
{
  for (size_t i=0; i < observation_times.size(); ++i)
    _movie_frames.push_back(new model_polarized_image_shearing_spot(start_observation, observation_times[i], bkgd_riaf, frequency,M,D));
}

model_polarized_movie_shearing_spot::~model_polarized_movie_shearing_spot()
{
  for (size_t i=0; i<_movie_frames.size(); ++i)
    delete _movie_frames[i];
}

void model_polarized_movie_shearing_spot::set_image_resolution(int Nray, int number_of_refines)
{
  for (size_t i=0; i<_movie_frames.size(); ++i)
    _movie_frames[i]->set_image_resolution(Nray, number_of_refines);
}

void model_polarized_movie_shearing_spot::set_screen_size(double Rmax)
{
  for (size_t i = 0; i < _movie_frames.size(); ++i)
    _movie_frames[i]->set_screen_size(Rmax);
}

  
};
