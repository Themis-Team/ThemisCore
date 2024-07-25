/*!
  \file model_image_upsample.cpp
  \author Avery Broderick
  \date  May, 2019
  \brief Implements the model_image_upsample image class.
  \details To be added
*/

#include "model_image_upsample.h"
#include <iostream>
#include <iomanip>

namespace Themis {

  model_image_upsample::model_image_upsample(model_image& image, size_t factor, std::string method)
    : _image(image), _factor(factor), _method(method)
  {
  }

  std::string model_image_upsample::model_tag() const
  {
    std::stringstream tag;
    tag << "model_image_upsample  " << _factor << " " << _method << '\n';
    tag << "SUBTAG START\n";
    tag << _image.model_tag() << '\n';
    tag << "SUBTAG FINISH";
    
    return tag.str();
  }

  void model_image_upsample::generate_model(std::vector<double> parameters)
  {
    // Check to see if these differ from last set used.
    if (_generated_model && parameters==_current_parameters)
      return;
    else
    {
      _current_parameters = parameters;
      
      // Generate the image using the user-supplied routine
      generate_image(parameters,_I,_alpha,_beta);
      
      // Set some boolean flags for what is and is not defined
      _generated_model = true;
      _generated_visibilities = false;
    }
  }

  void model_image_upsample::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    // Obtain the original image
    std::vector< std::vector<double> > alpha_orig_vector, beta_orig_vector, I_orig_vector;
    _image.generate_model(parameters);
    _image.get_image(alpha_orig_vector,beta_orig_vector,I_orig_vector);

    // Create interpolator object
    size_t Nx_orig=alpha_orig_vector.size();
    size_t Ny_orig=alpha_orig_vector[0].size();
    std::valarray<double> alpha_orig(Nx_orig), beta_orig(Ny_orig), I_orig(Nx_orig*Ny_orig);
    for (size_t j=0; j<Nx_orig; ++j)
      alpha_orig[j] = alpha_orig_vector[j][0];
    for (size_t k=0; k<Ny_orig; ++k)
      beta_orig[k] = beta_orig_vector[0][k];
    for (size_t j=0; j<Nx_orig; ++j)
      for (size_t k=0; k<Ny_orig; ++k)
	I_orig[k+Ny_orig*j] = I_orig_vector[j][k];
    _interp.set_f(alpha_orig,beta_orig,I_orig);
    _interp.use_forward_difference();
    /*
    /// DEBUGGING
    for (size_t j=0; j<Nx_orig; ++j)
      for (size_t k=0; k<Ny_orig; ++k)
      {
	double val;
	_interp.bicubic_spline(alpha_orig[j],beta_orig[k],val);
	std::cout << std::setw(15) << alpha_orig[j]
		  << std::setw(15) << beta_orig[k]
		  << std::setw(15) << I_orig[k+Ny_orig*j]
		  << std::setw(15) << val
		  << std::setw(15) << "miu:debg"
		  << std::endl;
      }
    std::cout << std::endl << std::endl;
    */

    // Generate up-sampled image
    size_t Nx = (Nx_orig-1) * _factor + 1;
    size_t Ny = (Ny_orig-1) * _factor + 1;
    if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(Nx))
    {
      alpha.resize(Nx);
      beta.resize(Nx);
      I.resize(Nx);
      for (size_t j=0; j<alpha.size(); j++)
      {
	if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(Ny))
        {
	  alpha[j].resize(Ny,0.0);
	  beta[j].resize(Ny,0.0);
	  I[j].resize(Ny,0.0);
	}
      }
    }
    {
      double dx = (alpha_orig_vector[Nx_orig-1][Ny_orig-1]-alpha_orig_vector[0][0])/(int(Nx)-1);
      double dy = (beta_orig_vector[Nx_orig-1][Ny_orig-1]-beta_orig_vector[0][0])/(int(Ny)-1);
    
      // Fill array with new image
      for (size_t j=0; j<alpha.size(); j++)
      {
	for (size_t k=0; k<alpha[j].size(); k++)
        {
	  alpha[j][k] = double(j)*dx + alpha_orig_vector[0][0];
	  beta[j][k] = double(k)*dy  + beta_orig_vector[0][0];
	}
      }

      double val;
      if (_method=="bicubic_spline")
      {
	for (size_t j=0; j<alpha.size(); j++)
	  for (size_t k=0; k<alpha[j].size(); k++)
	  {
	    _interp.bicubic_spline(alpha[j][k],beta[j][k],val);
	    I[j][k] = val;
	  }
      }
      else if (_method=="linear")
      {
	for (size_t j=0; j<alpha.size(); j++)
	  for (size_t k=0; k<alpha[j].size(); k++)
	  {
	    _interp.linear(alpha[j][k],beta[j][k],val);
	    I[j][k] = val;
	  }
      }
      else if (_method=="bicubic")
      {
	for (size_t j=0; j<alpha.size(); j++)
	  for (size_t k=0; k<alpha[j].size(); k++)
	  {
	    _interp.bicubic(alpha[j][k],beta[j][k],val);
	    I[j][k] = val;
	  }
      }
      else
      {
	std::cerr << "ERROR: model_image_upsample did not recognize method option " << _method << '\n';
	std::exit(1);
      }
    }
  }
};
