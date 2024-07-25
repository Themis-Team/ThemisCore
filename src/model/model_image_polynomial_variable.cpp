/*!
  \file model_image_polynomial_variable.cpp
  \author Avery Broderick
  \date  March, 2019
  \brief Implements the model_image_polynomial_variable class, which generates a variable model by permitting model parameters to vary with time.  All parameters are constrained to vary as polynomial functions of time, though the order of each may differ.  A reference time may be set.
  \details To be added
*/

#include "model_image_polynomial_variable.h"
#include "data_visibility.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <cmath>

namespace Themis {

  model_image_polynomial_variable::model_image_polynomial_variable(model_image& image, std::vector<int> orders, double tref)
    : _image(image), _orders(orders), _tref(tref)
  {
    if (_orders.size()!=_image.size()) 
    {
      std::cerr << "ERROR: model_image_polynomial_variable constructor needs order vector of equal size to parameters of image.\n";
      std::exit(1);
    }

    size_t size=0;
    for (size_t j=0; j<_image.size(); ++j)
      size += (1+_orders[j]);

    _image_parameters.resize(_image.size());

    _new_size = size;
  }

  model_image_polynomial_variable::model_image_polynomial_variable(model_image& image, int order, double tref)
    : _image(image), _orders(image.size(),order), _tref(tref)
  {
    size_t size=0;
    for (size_t j=0; j<_image.size(); ++j)
      size += (1+_orders[j]);

    _image_parameters.resize(_image.size());

    _new_size = size;
  }

  void model_image_polynomial_variable::set_reference_time(double tref)
  {
    _tref = tref;
  }

  double model_image_polynomial_variable::reference_time() const
  {
    return _tref;
  }
  
  std::string model_image_polynomial_variable::model_tag() const
  {
    std::stringstream tag;
    int origtagprec = tag.precision();
    tag << "model_image_polynomial_variable " << std::setprecision(16) << _tref << std::setprecision(origtagprec);
    for (size_t j=0; j<_image.size(); ++j)
      tag << " " << _orders[j];
    tag << '\n';
    tag << "SUBTAG START\n";
    tag << _image.model_tag() << '\n';
    tag << "SUBTAG FINISH";
    
    return tag.str();
  }

  
  void model_image_polynomial_variable::generate_model(std::vector<double> parameters)
  {
    _current_parameters = parameters;
  }

  void model_image_polynomial_variable::generate_image_model(double t)
  {
    double dt = t-_tref;
    for (size_t j=0,m=0; j<_image.size(); ++j)
    {
      _image_parameters[j] = 0.0;
      for (int k=0; k<=_orders[j]; ++k)
	_image_parameters[j] += _current_parameters[m++]*std::pow(dt,k);
    }

    // Generate the image model
    _image.generate_model(_image_parameters);
  }
 

  void model_image_polynomial_variable::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    std::cerr << "model_image_polynomial_variable::generate_image : This function is not implemented because no uniform image structure is specified at this time.\n";
    std::exit(1);
  } 




std::complex<double> model_image_polynomial_variable::visibility(datum_visibility& d, double acc)
  {
    generate_image_model(d.tJ2000);
    return _image.visibility(d,acc);
  }

  double model_image_polynomial_variable::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    generate_image_model(d.tJ2000);
    return _image.visibility_amplitude(d,acc);
  }

  double model_image_polynomial_variable::closure_phase(datum_closure_phase& d, double acc)
  {
    generate_image_model(d.tJ2000);
    return _image.closure_phase(d,acc);
  }



  double model_image_polynomial_variable::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    generate_image_model(d.tJ2000);
    return _image.closure_amplitude(d,acc);
  }

  
};
