/*!
  \file model_vrt2_library.h
  \author Avery Broderick
  \date  July, 2017
  \brief Header file for the model_image_vrt2_library image class.
  \details To be added
*/

#ifndef Themis_MODEL_VRT2_LIBRARY_H_
#define Themis_MODEL_VRT2_LIBRARY_H_

#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include "model_image_vrt2_pmap.h"
#include "vrt2.h"
#include <string>
#include <vector>

namespace Themis {

/*!
  \brief Defines an interface for reading in a vrt2 image library (i.e.,
  a collection of pmap files with a single index file describing the 
  parameters and indexing).

  \details Reads in a preexisting polarizmation map output file library.  The
  file read is that whose parameters are closest, as measured by the L2 norm, 
  to the parameter set passed.  Importantly, there is no attempt to interpolate
  between neighboring parameters values.

  Parameter list:\n
  - parameters[0] ... 1st library parameter listed in index file.
  - parameters[1] ... 2nd library parameter listed in index file.
  - parameters[N-2] . last library parameter listed in index file.
  - parameters[N-1] . Position angle (in model_image) in radians\n

  \warning Only implemented for up to 6-parameter libraries.  While this is trivially extendable, it is unlikely to be necessary.
*/
  class model_vrt2_library : public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude
  {
  public:
    //! Constructs a model_vrt2_library object.  Must take the file name for the index file and a template for the pmap file naming scheme, e.g., pmap_%03i.d, for a single parameter library.  Optionally takes a mass and distance (measured in cm), which otherwise default to those of Sgr A*.
    model_vrt2_library(std::string index_file_name, std::string pmap_file_format, size_t Nindex,
			     double Mcm=VRT2::VRT2_Constants::M_SgrA_cm, double Dcm=VRT2::VRT2_Constants::D_SgrA_cm);
    virtual ~model_vrt2_library() {};
    
    //! A user-supplied function that returns the number of the parameters the model expects
    virtual inline size_t size() const { return (_Nparams+1); };

    //! Reads in the pmap file whose parameters are closest to those specified and generates the underlying model_vrt2_pmap model.
    virtual void generate_model(std::vector<double> parameters);

    //! Returns visibility ampitudes in Jy computed from the image given a 
    //! datum_visibility_amplitude object, containing all of the accoutrements. 
    //! While this provides access to the actual data value, the two could 
    //! be separated if necessary.  Also takes an accuracy parameter with 
    //! the same units as the data, indicating the accuracy with which the 
    //! model must generate a comparison value.  Note that this is redefined 
    //! to accomodate the possibility of using the analytical computation.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double acc);
    
    /*! Returns closure phase in degrees computed from the image given a 
      datum_closure_phase object, containing all of the accoutrements. While 
      this provides access to the actual data value, the two could be separated 
      if necessary.  Also takes an accuracy parameter with the same units as 
      the data, indicating the accuracy with which the model must generate a 
      comparison value. Note that this is redefined since the closure phase of 
      Gaussian images is identically zero. */
    virtual double closure_phase(datum_closure_phase& d, double acc);
    
    /*! Returns closure amplitude computed from the image given a 
      datum_closure_phase object, containing all of the accoutrements.  While 
      this provides access to the actual data value, the two could be separated 
      if necessary.  Also takes an accuracy parameter with the same units as 
      the data, indicating the accuracy with which the model must generate a 
      comparison value.  Note that this can be redefined in child classes. */
    virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);

  
    //! Provides direct access to the constructed image.  Sets a 2D grid of angles (alpha, beta) in radians and intensities in Jy per steradian.
    void get_image(std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta, std::vector<std::vector<double> >& I) const;


    
  private:
    model_image_vrt2_pmap _model;
    
    const std::string _pmap_file_format;
    size_t _Nparams; //!< Number of NON-POSITION ANGLE parameters.
    
    std::vector<int> get_library_indexes(std::vector<double> params);
    
    std::vector< std::vector<int> > _index_list;
    std::vector< std::vector<double> > _parameter_list;
    
  };

};
#endif
