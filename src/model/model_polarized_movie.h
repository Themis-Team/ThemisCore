/*!
  \file model_polarized_movie.h
  \author Paul Tiede
  \date May, 2018
  \brief Header file for the model movie class
*/

#ifndef Themis_MODEL_POLARIZED_MOVIE_H_
#define Themis_MODEL_POLARIZED_MOVIE_H_

#include <vector>
#include <complex>
#include <fstream>

#include "model_crosshand_visibilities.h"
#include "model_polarized_image.h"
#include <iostream>
#include <mpi.h>

namespace Themis {

/*!
  \brief Defines the interface for models that generate movie data with a collection of utility functions that allow the user to compute visibility amplitudes, closure phases, etc. Also will accept a movie interpolator function that allows the user to specify how the frame of the movie is chosen.

  /details While many EHT models provide images time variability of thus movies are important when understanding accretion flow and spacetime dynamics. This class provides the general interphase for making movies whose models the are inherently time dependent, such as the model_image_orbiting_spot.h from Broderick & Loeb 2006. This is accomplished my creating a series of model_polarized_image.h clases at the times specified by the user. Like the model image class this class provides both an interface to interferometric data models and utility functions for computing the appropriate data types. Furthermore, the model expects the last parameter given to be the rotation agle of the image on the sky. 

  \warning This class contains multiple virtual functions making it impossible to generate and explicit instantiation. This also relies on the model_polarized_image.h functionality to compute the FFTW to generate the compex visibilities.
*/
template <class model_polarized_image>
class model_polarized_movie : public model_crosshand_visibilities, public model_polarization_fraction, public model_visibility, public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude
{
  public:
    model_polarized_movie(std::vector<double> observation_times);
    virtual ~model_polarized_movie();

    //! A user-supplied function that returns the number of the parameters the model expects
    virtual inline size_t size() const {return 1;};

    virtual void generate_model(std::vector<double> parameters);
    
    
    //! Returns a vector of complex visibility corresponding to RR,LL,RL,LR in Jy computed from the image given a datum_crosshand_visibilities_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual std::vector< std::complex<double> > crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy);

    //! Returns complex visibility in Jy computed from the image given a datum_visibility object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

    //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double closure_phase(datum_closure_phase& d, double accuracy);

    //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

    //! A user-supplied function that returns the closure amplitudes.  Takes a datum_polarization_fraction to provide access to the various accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    virtual double polarization_fraction(datum_polarization_fraction& d, double accuracy);

  //! Provides direct access to the constructed image.  Sets a 2D grid of angles (alpha, beta) in radians and intensities in Jy per pixel.
  void get_movie_frame(double t_frame, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;
  
  
  void get_movie_frame(double t_frame, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V) const;






  //! Write a unique identifying tag for use with the ThemisPy plotting features. This calls the overloaded version with the outstream, which is the only function that need be rewritten in child classes.
  void write_model_tag_file(std::string tagfilename="model_image.tag") const
  {
    std::ofstream tagout(tagfilename);
    write_model_tag_file(tagout);
  };

  //! Write a unique identifying tag for use with the ThemisPy plotting features. For most child classes, the default implementation is suffcient.  However, should that not be the case, this is the only function that need be rewritten in child classes.
  virtual void write_model_tag_file(std::ofstream& tagout) const
  {
    tagout << "tagvers-1.0\n" << model_tag() << std::endl;
  };

  //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features. This function SHOULD be defined in subsequent model_image classes with a unique identifier that contains sufficient information about the hyperparameters to uniquely determine the image.  By default it writes "UNDEFINED"
  virtual std::string model_tag() const
  {
    return "UNDEFINED";
  };


  //! Outputs the frame of the movie at time t to the file with name fname. If rotate is true then each frame will be rotated by the position angle.
  void output_movie_frame(double t_frame, std::string fname, bool rotate=false);
  
  
  //! Provides ability to use bicubic spline interpolator (true) instead of regular bicubic. Code defaults to false.
  void use_spline_interp( bool use_spline );

  //! Provides the ability to get frame times in the movie.
  inline std::vector<double> get_frame_times(){return _observation_times;};

  //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
  virtual void set_mpi_communicator(MPI_Comm comm) ;


  //! Provides access to the the individual model_polarized_image frames of the base movie. Requires the movie index.
  model_polarized_image* get_movie_model_frame(size_t i);
  
protected:
  bool _generated_model;
  
  std::vector<double> _observation_times;

  std::vector< model_polarized_image* > _movie_frames;

  std::vector<double> _parameters;

  virtual size_t find_time_index(double tobs) const;


};

/////////////////////////////////////////////////////////////
// Implmentation goes here since I am using a template, and there isn't a way to put this in a separate .cpp?
template <class model_polarized_image>
model_polarized_movie<model_polarized_image>::model_polarized_movie(std::vector<double> observation_times)
  : _generated_model(false), _observation_times(observation_times)
{
}

template <class model_polarized_image>
model_polarized_image* model_polarized_movie<model_polarized_image>::get_movie_model_frame(size_t i)
{
  if ( i > _movie_frames.size() ){
    std::cerr << "index i out of range for movie_frames\n";
    std::exit(1);
  }
  return _movie_frames[i];
};

template <class model_polarized_image>
model_polarized_movie<model_polarized_image>::~model_polarized_movie()
{
}

template <class model_polarized_image>
void model_polarized_movie<model_polarized_image>::set_mpi_communicator(MPI_Comm comm)
{
  for (size_t i=0; i<_movie_frames.size(); ++i)
    _movie_frames[i]->set_mpi_communicator(comm);
}

template <class model_polarized_image>
void model_polarized_movie<model_polarized_image>::generate_model(std::vector<double> parameters)
{
  if (_parameters == parameters && _generated_model)
    return;
  else
  {
    _parameters = parameters;
    _generated_model = true; 
  }
}

template <class model_polarized_image>
std::vector<std::complex<double> > model_polarized_movie<model_polarized_image>::crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->crosshand_visibilities(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_polarized_movie::crosshand_visibilities : must generate model and spot table before visibility_amplitude\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
}


template <class model_polarized_image>
std::complex<double> model_polarized_movie<model_polarized_image>::visibility(datum_visibility& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->visibility(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_polarized_movie::visibility : must generate model and spot table before visibility_amplitude\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
}


template <class model_polarized_image>
double model_polarized_movie<model_polarized_image>::polarization_fraction(datum_polarization_fraction& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->polarization_fraction(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_polarized_movie::polarization_fraction : must generate model and spot table before visibility_amplitude\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
}


template <class model_polarized_image>
double model_polarized_movie<model_polarized_image>::closure_phase(datum_closure_phase& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->closure_phase(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_polarized_movie::closure_phase : Must generate model and spot table before closure_phase\n"
              << "model generated: " << _generated_model << std::endl;
    std::exit(1);
  }
}
template <class model_polarized_image>
double model_polarized_movie<model_polarized_image>::visibility_amplitude(datum_visibility_amplitude& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->visibility_amplitude(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_polarized_movie::visibility_amplitude : must generate model and spot table before visibility_amplitude\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
}


template <class model_polarized_image>
double model_polarized_movie<model_polarized_image>::closure_amplitude(datum_closure_amplitude& d, double accuracy)
{
  size_t ii=find_time_index(d.tJ2000);
  if (_generated_model )
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->closure_amplitude(d,accuracy) );
  } 
  else 
  {
    std::cerr << "model_polarized_movie::closure_amplitude : must generate model and spot table before visibility_amplitude\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
}


template <class model_polarized_image>
size_t model_polarized_movie<model_polarized_image>::find_time_index(double tobs) const
{

  for ( size_t i = 1; i < _movie_frames.size(); i++)
  {
    if ( tobs < _observation_times[i] )
      return i-1;
  }
  return _movie_frames.size()-1;
}


/*
template <class model_polarized_image>
size_t model_polarized_movie<model_polarized_image>::find_time_index(double tobs) const
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
template <class model_polarized_image>
void model_polarized_movie<model_polarized_image>::get_movie_frame(double t_frame, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const
{
  size_t ii=find_time_index(t_frame);
  if (_generated_model)
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->get_image(alpha,beta,I) );
  } 
  else 
  {
    std::cerr << "model_polarized_movie::get_movie_frame : Must generate model before getting movie frame\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
 
}


template <class model_polarized_image>
void model_polarized_movie<model_polarized_image>::get_movie_frame(double t_frame, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V) const
{
  size_t ii=find_time_index(t_frame);
  if (_generated_model)
  {
    _movie_frames[ii]->generate_model(_parameters);
    return ( _movie_frames[ii]->get_image(alpha,beta,I, Q, U, V) );
  } 
  else 
  {
    std::cerr << "model_polarized_movie::get_movie_frame : Must generate model before getting movie frame\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
 
}


template <class model_polarized_image>
void model_polarized_movie<model_polarized_image>::output_movie_frame(double t_frame, std::string fname, bool rotate)
{
  size_t ii=find_time_index(t_frame);
  if (_generated_model)
  {
    _movie_frames[ii]->generate_model(_parameters);
    _movie_frames[ii]->output_image(fname, rotate);
  } 
  else 
  {
    std::cerr << "model_polarized_movie::get_movie_frame : Must generate model before getting movie frame\n"
              << "generated model: " << _generated_model << std::endl;
    std::exit(1);
  }
 
}
  
template <class model_polarized_image>
void model_polarized_movie<model_polarized_image>::use_spline_interp(bool use_spline)
{
  for ( size_t i = 0; i < _movie_frames.size(); ++i)
    _movie_frames[i]->use_spline_interp(use_spline);
}



};

#endif
