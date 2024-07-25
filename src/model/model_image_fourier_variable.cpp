/*!
  \file model_image_fourier_variable.cpp
  \author Avery Broderick
  \date  March, 2019
  \brief Implements the model_image_fourier_variable class, which generates a variable model by permitting model parameters to vary with time.  All parameters are constrained to vary as fourier functions of time, though the order of each may differ.  A reference time may be set.
  \details To be added
*/

#include "model_image_fourier_variable.h"
#include "data_visibility.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <cmath>

namespace Themis {

  model_image_fourier_variable::model_image_fourier_variable(model_image& image, std::vector<int> orders, double tstart, double tend)
    : _image(image), _orders(orders), _tstart(tstart), _tend(tend)
  {
    if (_orders.size()!=_image.size()) 
    {
      std::cerr << "ERROR: model_image_fourier_variable constructor needs order vector of equal size to parameters of image.\n";
      std::exit(1);
    }

    size_t size=0;
    for (size_t j=0; j<_image.size(); ++j)
      size += (1+2*_orders[j]);

    _image_parameters.resize(_image.size());

    _new_size = size;

    _prior_ptrs.resize(0);
  }

  model_image_fourier_variable::model_image_fourier_variable(model_image& image, int order, double tstart, double tend)
    : _image(image), _orders(image.size(),order), _tstart(tstart), _tend(tend)
  {
    size_t size=0;
    for (size_t j=0; j<_image.size(); ++j)
      size += (1+2*_orders[j]);

    _image_parameters.resize(_image.size());

    _new_size = size;

    _prior_ptrs.resize(0);
  }

  void model_image_fourier_variable::set_start_time(double tref)
  {
    _tstart = tref;
  }

  void model_image_fourier_variable::set_end_time(double tref)
  {
    _tend = tref;
  }

  double model_image_fourier_variable::start_time() const
  {
    return _tstart;
  }

  double model_image_fourier_variable::end_time() const
  {
    return _tend;
  }

  void model_image_fourier_variable::set_priors(std::vector<prior_base*> priors)
  {
    _prior_ptrs = priors;
  }
  
  std::string model_image_fourier_variable::model_tag() const
  {
    std::stringstream tag;
    int origtagprec = tag.precision();
    tag << "model_image_fourier_variable " << std::setprecision(16) << _tstart << " " << std::setprecision(16) << _tend << std::setprecision(origtagprec);
    for (size_t j=0; j<_image.size(); ++j)
      tag << " " << _orders[j];
    tag << '\n';
    tag << "SUBTAG START\n";
    tag << _image.model_tag() << '\n';
    tag << "SUBTAG FINISH";
    
    return tag.str();
  }

  
  void model_image_fourier_variable::generate_model(std::vector<double> parameters)
  {
    _current_parameters = parameters;
  }

#define TINY (1e-14)
  void model_image_fourier_variable::generate_image_model(double t)
  {
    double dt = M_PI*(t-_tstart)/(_tend-_tstart);
    for (size_t j=0,m=0; j<_image.size(); ++j)
    {
      _image_parameters[j] = _current_parameters[m++];
      for (int k=0; k<_orders[j]; ++k)
      {
	_image_parameters[j] += _current_parameters[m++]*std::cos(dt*k);
	_image_parameters[j] += _current_parameters[m++]*std::sin(dt*k);
      }
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Prior check?  CHECK THAT THIS DOESN'T CREATE PROBLEMS IN PRACTICE DUE TO INFINITE BOUNDS.
    if (_prior_ptrs.size()>0)
      for (size_t j=0; j<_image.size(); ++j)
      {
	double small_shift = TINY*0.5*( _prior_ptrs[j]->lower_bound() +  _prior_ptrs[j]->upper_bound() );
        //if (rank == 0)
        //std::cerr << "Image Parameter before" 
        //          << j << std::setw(15) << _image_parameters[j] << std::endl;
	_image_parameters[j] = std::min( std::max( _image_parameters[j], _prior_ptrs[j]->lower_bound()+small_shift ), _prior_ptrs[j]->upper_bound()-small_shift );
        //if (rank == 0)
        //std::cerr << "Image Parameter after" 
        //          << j << std::setw(15) << _image_parameters[j] << std::endl;
                 
      }
    
    // Generate the image model
    _image.generate_model(_image_parameters);
  }
#undef TINY
 

  void model_image_fourier_variable::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    std::cerr << "model_image_fourier_variable::generate_image : This function is not implemented because no uniform image structure is specified at this time.\n";
    std::exit(1);
  } 




std::complex<double> model_image_fourier_variable::visibility(datum_visibility& d, double acc)
  {
    generate_image_model(d.tJ2000);
    return _image.visibility(d,acc);
  }

  double model_image_fourier_variable::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    generate_image_model(d.tJ2000);
    return _image.visibility_amplitude(d,acc);
  }

  double model_image_fourier_variable::closure_phase(datum_closure_phase& d, double acc)
  {
    generate_image_model(d.tJ2000);
    return _image.closure_phase(d,acc);
  }



  double model_image_fourier_variable::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    generate_image_model(d.tJ2000);
    return _image.closure_amplitude(d,acc);
  }

  
};
