/*!
  \file model_polarized_image_sum.cpp
  \author Avery Broderick
  \date  November, 2018
  \brief Implements the model_polarized_image_sum class, which sums different model images, generating a convenient way to create multi-component image models from single image components.
  \details To be added
*/

#include "model_polarized_image_sum.h"
#include "data_visibility.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <cmath>

namespace Themis {

  model_polarized_image_sum::model_polarized_image_sum( std::vector< model_polarized_image* > images, std::string offset_coordinates)
    : _images(images), _offset_coordinates(offset_coordinates)
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_polarized_image_sum in rank " << world_rank << std::endl;

    _size=0;
    for (size_t i=0; i<_images.size(); ++i)
      _size += _images[i]->size()+2;

    _x.resize(_images.size(),0.0);
    _y.resize(_images.size(),0.0);
  }

  model_polarized_image_sum::model_polarized_image_sum(std::string offset_coordinates)
    : _offset_coordinates(offset_coordinates)    
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::cout << "Creating model_polarized_image_sum in rank " << world_rank << std::endl;
    
    _size=0;
    _images.resize(0);
    _x.resize(0);
    _y.resize(0);
  }

  
  void model_polarized_image_sum::add_model_polarized_image(model_polarized_image& image)
  {
    _images.push_back(&image);
    _size += image.size()+2;

    _x.push_back(0.0);
    _y.push_back(0.0);
  }

  void model_polarized_image_sum::generate_model(std::vector<double> parameters)
  {
    // Check to see if these differ from last set used.
    if (_generated_model && parameters==_current_parameters)
      return;
    else
    {
      _current_parameters = parameters;

      // Read and strip off Dterm parameters
      read_and_strip_Dterm_parameters(parameters);

      std::vector<double> psub;
      for (size_t i=0,k=0; i<_images.size(); ++i)
      {
	// Generate a list of the parameters for the ith image model
	psub.resize(_images[i]->size());
	for (size_t j=0; j<_images[i]->size(); ++j,++k)
	  psub[j] = parameters[k];
	// Save the offset in x and y
	if (_offset_coordinates=="Cartesian")
	{
	  _x[i] = parameters[k++];
	  _y[i] = parameters[k++];
	}
	else if (_offset_coordinates=="polar")
	{
	  double r = parameters[k++];
	  double theta = parameters[k++];
	  _x[i] = r*std::cos(theta);
	  _y[i] = r*std::sin(theta);
	}
	else
	{
	  std::cerr << "ERROR: Unrecognized coordinate system in model_image_sum.\n";
	  std::exit(1);
	}
	// Generate the ith image model.
	_images[i]->generate_model(psub);
      }
      
      // Set some boolean flags for what is and is not defined
      _generated_model = true;
      _generated_visibilities = false;
    }
  }
 

  void model_polarized_image_sum::generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    std::cerr << "ERROR: model_polarized_image_sum::generate_polarized_image : This function is not implemented because no uniform image structure is specified at this time.\n";
    std::exit(1);
  } 

  void model_polarized_image_sum::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
  {
    std::cerr << "ERROR: model_polarized_image_sum::generate_image : This function is not implemented because no uniform image structure is specified at this time.\n";
    std::exit(1);
  } 

  std::string model_polarized_image_sum::model_tag() const
  {
    std::stringstream tag;

    tag << "model_polarized_image_sum " << _offset_coordinates
	<< " " << _modeling_Dterms;
    if (_modeling_Dterms)
      for (size_t j=0; j<_station_codes.size(); ++j)
	tag << " " << _station_codes[j];
    tag << "\n";
    tag << "SUBTAG START\n";
    for (size_t j=0; j<_images.size(); ++j)
      tag << _images[j]->model_tag() << '\n';
    tag << "SUBTAG FINISH";
    
    return tag.str();
  }
  
  std::vector< std::complex<double> > model_polarized_image_sum::crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy)
  {
    const std::complex<double> i(0.0,1.0);
    std::complex<double> phase_factor;
    std::vector< std::complex<double> > cvo;
    std::vector< std::complex<double> > crosshand_vector(4,std::complex<double>(0.0,0.0));

    for (size_t j=0; j<_images.size(); ++j)
    {
      phase_factor =  std::exp( - 2.0*M_PI* i * (_x[j]*(-d.u) + _y[j]*d.v) );
      cvo = _images[j]->crosshand_visibilities(d,accuracy);
      for (size_t k=0; k<4; ++k) 
	crosshand_vector[k] = crosshand_vector[k] + phase_factor*cvo[k];
    }

    // Apply Dterms
    apply_Dterms(d,crosshand_vector);

    return ( crosshand_vector );
  }  


  std::complex<double> model_polarized_image_sum::visibility(datum_visibility& d, double acc)
  {
    const std::complex<double> i(0.0,1.0);
    std::complex<double> exponent;
    std::complex<double> V(0.,0.);
      
    for (size_t j=0; j<_images.size(); ++j)
    {
      exponent =  - 2.0*M_PI* i * (_x[j]*(-d.u) + _y[j]*d.v);
      V +=  std::exp(exponent) * _images[j]->visibility(d,acc);
    }

    return ( V );
  }

  double model_polarized_image_sum::visibility_amplitude(datum_visibility_amplitude& d, double acc)
  {
    datum_visibility tmp(d.u,d.v,std::complex<double>(d.V,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);

    return ( std::abs(visibility(tmp,acc)) );
  }

  double model_polarized_image_sum::closure_phase(datum_closure_phase& d, double acc)
  {
    datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
    datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station1,d.Source);
    std::complex<double> V123 = visibility(tmp1,acc)*visibility(tmp2,acc)*visibility(tmp3,acc);

    return ( std::imag(std::log(V123))*180.0/M_PI );
  }



  double model_polarized_image_sum::closure_amplitude(datum_closure_amplitude& d, double acc)
  {
    datum_visibility tmp1(d.u1,d.v1,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station1,d.Station2,d.Source);
    datum_visibility tmp2(d.u2,d.v2,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station2,d.Station3,d.Source);
    datum_visibility tmp3(d.u3,d.v3,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station3,d.Station4,d.Source);
    datum_visibility tmp4(d.u4,d.v4,std::complex<double>(0,0),std::complex<double>(d.err,d.err),d.frequency,d.tJ2000,d.Station4,d.Station1,d.Source);

    double V1234 = std::abs( (visibility(tmp1,acc)*visibility(tmp3,acc)) / (visibility(tmp2,acc)*visibility(tmp4,acc)) );
      
    return ( V1234 );
  }

  
};
