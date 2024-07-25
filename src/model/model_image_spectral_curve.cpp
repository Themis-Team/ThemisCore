/*!
  \file model_image_spectral_curve.cpp
  \author Avery Broderick
  \date  March, 2019
  \brief Implements the model_image_spectral_curve class, which generates a variable model by permitting model parameters to vary with time.  All parameters are constrained to vary as polynomial functions of time, though the order of each may differ.  A reference time may be set.
  \details To be added
*/

#include "model_image_spectral_curve.h"
#include "data_visibility.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <cmath>

namespace Themis {

  model_image_spectral_curve::model_image_spectral_curve(model_image& image, std::vector<int> orders, double freqref)
    : _image(image), _orders(orders), _freqref(freqref)
  {
    if (_orders.size()!=_image.size()) 
    {
      std::cerr << "ERROR: model_image_spectral_curve constructor needs order vector of equal size to parameters of image.\n";
      std::exit(1);
    }

    size_t size=0;
    for (size_t j=0; j<_image.size(); ++j)
      size += (1+_orders[j]);

    _image_parameters.resize(_image.size());

    _new_size = size;
  }

  model_image_spectral_curve::model_image_spectral_curve(model_image& image, int order, double freqref)
    : _image(image), _orders(image.size(),order), _freqref(freqref)
  {
    size_t size=0;
    for (size_t j=0; j<_image.size(); ++j)
      size += (1+_orders[j]);

    _image_parameters.resize(_image.size());

    _new_size = size;
  }

  void model_image_spectral_curve::set_reference_freq(double freqref)
  {
    _freqref = freqref;
  }

  double model_image_spectral_curve::reference_freq() const
  {
    return _freqref;
  }
  
  std::string model_image_spectral_curve::model_tag() const
  {
    std::stringstream tag;
    int origtagprec = tag.precision();
    tag << "model_image_spectral_curve " << std::setprecision(16) << _freqref << std::setprecision(origtagprec);
    for (size_t j=0; j<_image.size(); ++j)
      tag << " " << _orders[j];
    tag << '\n';
    tag << "SUBTAG START\n";
    tag << _image.model_tag() << '\n';
    tag << "SUBTAG FINISH";
    
    return tag.str();
  }

  
  void model_image_spectral_curve::generate_model(std::vector<double> parameters)
  {
    _current_parameters = parameters;
  }

  void model_image_spectral_curve::generate_image_model(double freq)
  {
    double df = std::log(freq/_freqref);
    for (size_t j=0,m=0; j<_image.size(); ++j)
    {
      _image_parameters[j] = _current_parameters[m++];
      double dl = 0.0;
      for (int k=1; k<=_orders[j]; ++k){
	dl += _current_parameters[m++]*std::pow(df,k); 
      }
      _image_parameters[j] += dl;
      //std::cout << _image_parameters[j] << std::endl;
    }
    //std::cout << df << std::endl;
    // Generate the image model
    _image.generate_model(_image_parameters);
  }
 

  void model_image_spectral_curve::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    std::cerr << "model_image_spectral_curve::generate_image : This function is not implemented because no uniform image structure is specified at this time.\n";
    std::exit(1);
  } 




std::complex<double> model_image_spectral_curve::visibility(datum_visibility& d, double acc)
  {
    generate_image_model(d.frequency);
    return _image.visibility(d,acc);
  }

  double model_image_spectral_curve::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    generate_image_model(d.frequency);
    return _image.visibility_amplitude(d,acc);
  }

  double model_image_spectral_curve::closure_phase(datum_closure_phase& d, double acc)
  {
    generate_image_model(d.frequency);
    return _image.closure_phase(d,acc);
  }



  double model_image_spectral_curve::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    generate_image_model(d.frequency);
    return _image.closure_amplitude(d,acc);
  }

  
};
