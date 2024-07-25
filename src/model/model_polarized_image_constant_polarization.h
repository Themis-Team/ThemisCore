/*!
  \file model_polarized_image_constant_polarization.h
  \author Avery E. Broderick
  \date March, 2020
  \brief Header file for a polarized image model class with a constant polarization applied to a model image.
*/

#ifndef Themis_MODEL_POLARIZED_IMAGE_CONSTANT_POLARIZATION_H_
#define Themis_MODEL_POLARIZED_IMAGE_CONSTANT_POLARIZATION_H_

#include <vector>
#include <complex>

#include "model_polarized_image.h"
#include "model_image.h"

#include <mpi.h>

namespace Themis {

  /*!
    \brief Defines a class that takes a model_image and applies a constant polarization, returning a model_polarized_image instance.

    \details Takes a model_image object which defines the Stokes I map model and applies a constant, uniform polarization fraction and orientation.  

    Parameter list:\n
    - parameters[0] ... model_image object parameters
    - parameters[N] ... model_image object parameters
    - parameters[N+1] . Polarization fraction
    - parameters[N+2] . Polarization EVPA
    - parameters[N+3] . Polarization ellipticity (i.e., \f$ \tan[V/\sqrt(Q^2+U^2) \f$]

    Note that the polarization angle is stripped off of the end as we assume that it is included in the underlying model_image object.

    \warning If fitting the crosshand visibilities, you may need to make sure that the image has the shift freedom (i.e., offset in x,y).  This may be created from a normal model_image using model_image_sum.

    \todo
  */
  class model_polarized_image_constant_polarization : public model_polarized_image
  {
   public:
    model_polarized_image_constant_polarization(model_image& intensity_model);
    virtual ~model_polarized_image_constant_polarization();

    
  private:

    //! A user-supplied function that generates and returns rectalinear grid of intensities in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.  Note that it will be assumed that alpha and beta are defined as the image appears on the sky, e.g., beta running from S to N and alpha running from E to W.
    virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);

    //! A user-supplied function that generates and returns rectalinear grid of Stokes parameters in Jy/str located at pixels centered on angular positions alpha and beta, both specified in radians and aligned with a fiducial direction.  Note that the parameter vector has had the position removed.  Note that it will be assumed that alpha and beta are defined as the image appears on the sky, e.g., beta running from S to N and alpha running from E to W.
    virtual void generate_polarized_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& Q, std::vector<std::vector<double> >& U, std::vector<std::vector<double> >& V, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);
    
    //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
    virtual void generate_model(std::vector<double> parameters);
  
    //! Returns a vector of complex visibility corresponding to RR,LL,RL,LR in Jy computed from the image given a datum_crosshand_visibilities_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    virtual std::vector< std::complex<double> > crosshand_visibilities(datum_crosshand_visibilities& d, double accuracy);

    //! Returns complex visibility in Jy computed from the image given a datum_visibility object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    virtual std::complex<double> visibility(datum_visibility& d, double accuracy);

    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);

    //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    virtual double closure_phase(datum_closure_phase& d, double accuracy);

    //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

    //! A user-supplied function that returns the closure amplitudes.  Takes a datum_fractional_polarization to provide access to the various accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.
    //virtual double polarization_fraction(datum_polarization_fraction& d, double accuracy);

    //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
    virtual void set_mpi_communicator(MPI_Comm comm);

    //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features. This function SHOULD be defined in subsequent model_image classes with a unique identifier that contains sufficient information about the hyperparameters to uniquely determine the image.  By default it writes "UNDEFINED"
    virtual std::string model_tag() const;

    
   private:
    model_image& _intensity_model;
    double _polarization_fraction, _polarization_EVPA, _polarization_mu;

  };
}
#endif
