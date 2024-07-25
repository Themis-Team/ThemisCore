/*!
  \file model_movie_orbiting_spot.cpp
  \author Paul Tiede
  \date  April, 2018
  \brief Implements orbiting spot movie class.
*/

#include "model_movie_orbiting_spot.h"
#include <cmath>

namespace Themis {

model_movie_orbiting_spot::model_movie_orbiting_spot(double start_observation, std::vector<double> observation_times, std::string sed_fit_parameter_file, double frequency, double M, double D)
  :model_movie<model_image_orbiting_spot>(observation_times)
{
  for (size_t i=0; i < observation_times.size(); ++i)
    _movie_frames.push_back(new model_image_orbiting_spot(start_observation, observation_times[i], sed_fit_parameter_file, frequency,M,D));
}

model_movie_orbiting_spot::~model_movie_orbiting_spot()
{
  for (size_t i=0; i<_movie_frames.size(); ++i)
    delete _movie_frames[i];
}

void model_movie_orbiting_spot::set_image_resolution(int Nray)
{
  for (size_t i=0; i<_movie_frames.size(); ++i)
    _movie_frames[i]->set_image_resolution(Nray);
}

void model_movie_orbiting_spot::set_screen_size(double Rmax)
{
	for (size_t i = 0; i < _movie_frames.size(); ++i)
		_movie_frames[i]->set_screen_size(Rmax);
}

void model_movie_orbiting_spot::add_background_riaf()
{
  for ( size_t i = 0; i < _movie_frames.size(); ++i)
    _movie_frames[i]->add_background_riaf();
}
  
};
