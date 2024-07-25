/*!
  \file model_movie.h
  \author Paul Tiede
  \date May, 2018
  \brief Header file for the model movie class
*/

#ifndef Themis_MODEL_MOVIE_H_
#define Themis_MODEL_MOVIE_H_

#include <vector>
#include <complex>

#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "model_image.h"
#include <iostream>
#include <mpi.h>

namespace Themis {

/*!
  \brief Defines the interface for models that generate movie data with a collection of utility functions that allow the user to compute visibility amplitudes, closure phases, etc. Also will accept a movie interpolator function that allows the user to specify how the frame of the movie is chosen.

  /details While many EHT models provide images time variability of thus movies are important when understanding accretion flow and spacetime dynamics. This class provides the general interphase for making movies whose models the are inherently time dependent, such as the model_image_orbiting_spot.h from Broderick & Loeb 2006. This is accomplished my creating a series of model_image.h clases at the times specified by the user. Like the model image class this class provides both an interface to interferometric data models and utility functions for computing the appropriate data types. Furthermore, the model expects the last parameter given to be the rotation agle of the image on the sky. 

  \warning This class contains multiple virtual functions making it impossible to generate and explicit instantiation. This also relies on the model_image.h functionality to compute the FFTW to generate the compex visibilities.
*/
template <class model_image>
class model_movie : public model_visibility, public model_visibility_amplitude, public model_closure_phase//, public model_closure_amplitude
{
  public:
    model_movie(std::vector<double> observation_times);
    virtual ~model_movie();

    //! A user-supplied function that returns the number of the parameters the model expects
    virtual inline size_t size() const {return 1;};

    virtual void generate_model(std::vector<double> parameters);
    
    //! Returns visibility in Jy, computed numerically. The accuracy parameters isn't used presently.
    virtual std::complex<double> visibility(datum_visibility& d, double accuracy);
    //! Returns visibility amplitude in Jy, computed numerically. The accuracy parameters isn't used presently.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

    //! Returns closure ampitudes, computed numerically. The accuracy parameter is not used at present.
    virtual double closure_phase(datum_closure_phase& d, double accuracy);

    //virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

  //! Provides direct access to the constructed image.  Sets a 2D grid of angles (alpha, beta) in radians and intensities in Jy per pixel.
  void get_movie_frame(double t_frame, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;

  //! Provides direct access to the complex visibilities.  Sets a 2D grid of baselines (u,v) in lambda, and visibilites in Jy.
  void get_visibilities(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<std::complex<double> > >& V) const;

  //! Provides direct access to the visibility amplitudes.  Sets a 2D grid of baselines (u,v) in lambda, and visibilites in Jy.
  void get_visibility_amplitudes(std::vector<std::vector<double> >& u, std::vector<std::vector<double> >& v, std::vector<std::vector<double> >& V) const;

  //! Outputs the frame of the movie at time t to the file with name fname. If rotate is true then each frame will be rotated by the position angle.
  void output_movie_frame(double t_frame, std::string fname, bool rotate=false);
  
  
  //! Provides ability to use bicubic spline interpolator (true) instead of regular bicubic. Code defaults to false.
  void use_spline_interp( bool use_spline );

  //! Provides the ability to get frame times in the movie.
  inline std::vector<double> get_frame_times(){return _observation_times;};

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) ;


  //! Provides access to the the individual model_image frames of the base movie. Requires the movie index.
  model_image* get_movie_model_frame(size_t i);
  
protected:
  bool _generated_model;
  
  std::vector<double> _observation_times;

  std::vector< model_image* > _movie_frames;

  std::vector<double> _parameters;

  virtual size_t find_time_index(double tobs) const;


};

/////////////////////////////////////////////////////////////
// Implmentation goes here since I am using a template, and there isn't a way to put this in a separate .cpp?
template <class model_image>
model_movie<model_image>::model_movie(std::vector<double> observation_times)
  : _generated_model(false), _observation_times(observation_times)
{
}

template <class model_image>
model_image* model_movie<model_image>::get_movie_model_frame(size_t i)
{
  if ( i > _movie_frames.size() ){
    std::cerr << "index i out of range for movie_frames\n";
    std::exit(1);
  }
  return _movie_frames[i];
};

template <class model_image>
model_movie<model_image>::~model_movie()
{
}

template <class model_image>
void model_movie<model_image>::set_mpi_communicator(MPI_Comm comm)
{
  for (size_t i=0; i<_movie_frames.size(); ++i)
    _movie_frames[i]->set_mpi_communicator(comm);
}

template <class model_image>
void model_movie<model_image>::generate_model(std::vector<double> parameters)
{
  if (_parameters == parameters && _generated_model)
    return;
  else
  {
    _parameters = parameters;
    _generated_model = true; 
  }
}

template <class model_image>
std::complex<double> model_movie<model_image>::visibility(datum_visibility& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->visibility(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_movie::visibility : must generate model and spot table before visibility_amplitude\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
}

template <class model_image>
double model_movie<model_image>::closure_phase(datum_closure_phase& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->closure_phase(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_movie::closure_phase : Must generate model and spot table before closure_phase\n"
              << "model generated: " << _generated_model << std::endl;
    std::exit(1);
  }
}
template <class model_image>
double model_movie<model_image>::visibility_amplitude(datum_visibility_amplitude& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->visibility_amplitude(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_movie::visibility_amplitude : must generate model and spot table before visibility_amplitude\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
}


template <class model_image>
size_t model_movie<model_image>::find_time_index(double tobs) const
{

  for ( size_t i = 1; i < _movie_frames.size(); i++)
  {
    if ( tobs < _observation_times[i] )
      return i-1;
  }
  return _movie_frames.size()-1;
}


/*
template <class model_image>
size_t model_movie<model_image>::find_time_index(double tobs) const
{
  size_t imin=0;
  double dfmin=0;
  for (size_t i=0; i<_movie_frames.size(); ++i)
  {
    double df = std::fabs(std::log(tobs/_observation_times[i]));
    if (i==0 || df<dfmin)
    {
      imin=i;
      dfmin=df;
    }
  }
  return imin;
}
*/
template <class model_image>
void model_movie<model_image>::get_movie_frame(double t_frame, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{
  size_t ii=find_time_index(t_frame);
  if (_generated_model)
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->get_image(alpha,beta,I) );
  } 
  else 
  {
    std::cerr << "model_movie::get_movie_frame : Must generate model before getting movie frame\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
 
}


template <class model_image>
void model_movie<model_image>::output_movie_frame(double t_frame, std::string fname, bool rotate)
{
  size_t ii=find_time_index(t_frame);
  if (_generated_model)
  {
    _movie_frames[ii]->generate_model(_parameters);
    _movie_frames[ii]->output_image(fname, rotate);
  } 
  else 
  {
    std::cerr << "model_movie::get_movie_frame : Must generate model before getting movie frame\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
 
}
  
template <class model_image>
void model_movie<model_image>::use_spline_interp(bool use_spline)
{
  for ( size_t i = 0; i < _movie_frames.size(); ++i)
    _movie_frames[i]->use_spline_interp(use_spline);
}



};

#endif
