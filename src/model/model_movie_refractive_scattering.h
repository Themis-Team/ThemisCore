/*!
  \file model_movie_refractive_scattering.h
  \author Paul Tiede
  \date  May, 2019
  \brief Header file for refractively scattering a model_movie using model_image_refractive_scattering
  \details Creates a series of scattered images, using model_image_refractive_scattering at the times specified at model_movie passed at the beginning 
*/

#ifndef Themis_MODEL_MOVIE_REFRACTIVE_SCATTERING_H_
#define Themis_MODEL_MOVIE_REFRACTIVE_SCATTERING_H_

#include "model_movie.h"
#include "model_image_refractive_scattering.h"

#include <string>
#include <vector>

#include <mpi.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

namespace Themis {

  /*!
    \class model_image_refractive_scattering
    \author Paul Tiede
    \date Oct. 2018
    \brief Defines the interface for models that generates refractive scattered movies (it is templated off the model image class that you will scatter)
    \details Scattering implementation assumes we are in average strong-scattering regime where
    diffractive scintillation has been averaged over.
    Parameter list: \n
    - parameters[0...n-1]   ... Model parameters for the source sans position angle, i.e. if a RIAF then spin inclination and so on. 
    - parameters[n]         ... position angle for the images.
    - parameters[n+1...m]   ... nModes^2-1=m-n-1 normalized Fourier modes of the scattering screen where nModes is passed in the constructor.
                            ... note the normalized means the distribution of these modes is given by a Gaussian with zero mean and unit variance.

  */
template <class T>
class model_movie_refractive_scattering : public model_movie<model_image_refractive_scattering>
{
 public:
 
  /*!
    Constructor takes in a model_movie class and scatters it. Note that observation times are found from the model.
    For a description about what each option does see model_image_refractive_scattering
  */
  model_movie_refractive_scattering(T& model, size_t nModes, double tstart, 
                                  double frequency=230e9,
                                  std::string scattering_model="dipole",
                                  double observer_screen_distance=2.82*3.086e21, double source_screen_distance=5.53*3.086e21,
                                  double theta_maj_mas_cm=1.38, double theta_min_ma_cm=0.703, double POS_ANG=81.9, 
                                  double scatt_alpha=1.38, double r_in=800e5, double r_out=1e20,
                                  double vs_ss_kms = 50.0, double vy_ss_kms = 0.0);
  ~model_movie_refractive_scattering();

  //! Returns the number of the parameters the model expects
  virtual inline size_t size() const { return _model->size() + _nModes*_nModes-1; };

  
  //! Sets the image scattered image resolution to be used
  //! resolution of the image is nrayxnray, the default is 128,128 which is probably too much
  void set_image_resolution(size_t nray);

  //! Sets the fov size of the image in units of radians
  //! The current default is 100uas.
  void set_screen_size(double fov);
   

  /*
    \brief Provide access to ensemble average movie at time t_frame.
      
    \details Provide access to *ensemble average* image.
    \param time of movie in seconds (tJ2000 time)
    \param alpha coordinate 1 in image plane
    \param beta coordinate 2 in image plane
    \param I Intensity in Jy
  */
  void get_ensemble_average_frame(double t_frame, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;

 private:
  T* _model;
  size_t _nModes;
  double _tstart;
  
};



//--------------------------------------- Template declarations -----------------------------------//
template <class T>
model_movie_refractive_scattering<T>::model_movie_refractive_scattering(T& model, size_t nModes, 
                                  double tstart, double frequency,
                                  std::string scattering_model,
                                  double observer_screen_distance, double source_screen_distance,
                                  double theta_maj_mas_cm, double theta_min_ma_cm, double POS_ANG, 
                                  double scatt_alpha, double r_in, double r_out,
                                  double vx_ss_kms, double vy_ss_kms)
:model_movie<model_image_refractive_scattering>(model.get_frame_times()),_model(&model), _nModes(nModes), _tstart(tstart)
{
  for (size_t i=0; i < _observation_times.size(); ++i)
    _movie_frames.push_back(new model_image_refractive_scattering(*_model->get_movie_model_frame(i), _nModes, _observation_times[i]-_tstart, frequency, scattering_model, observer_screen_distance, source_screen_distance, theta_maj_mas_cm, theta_min_ma_cm, POS_ANG, scatt_alpha, r_in, r_out, vx_ss_kms, vy_ss_kms));
}

template <class T>
model_movie_refractive_scattering<T>::~model_movie_refractive_scattering()
{
  for (size_t i=0; i<_movie_frames.size(); ++i)
    delete _movie_frames[i];
}

template <class T>
void model_movie_refractive_scattering<T>::set_image_resolution(size_t Nray)
{
  for (size_t i=0; i<_movie_frames.size(); ++i)
    _movie_frames[i]->set_image_resolution(Nray);
}

template <class T>
void model_movie_refractive_scattering<T>::set_screen_size(double fov)
{
  for (size_t i = 0; i < _movie_frames.size(); ++i)
    _movie_frames[i]->set_screen_size(fov);
}

template <class T>
void model_movie_refractive_scattering<T>::get_ensemble_average_frame(double t_frame, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{
  size_t ii = find_time_index(t_frame); 

  if (_generated_model)
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->get_ensemble_average_image(alpha,beta,I) );
  } 
  else 
  {
    std::cerr << "model_movie::get_movie_frame : Must generate model before getting movie frame\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
}

  

};

#endif


