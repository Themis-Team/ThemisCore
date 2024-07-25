/*!
  \file model_ring.h
  \author Jorge A. Preciado
  \date  February 2018
  \brief Header file for the ring model class.
  \details To be added
*/

#ifndef Themis_MODEL_RING_H_
#define Themis_MODEL_RING_H_

#include "model_visibility_amplitude.h"
#include "model_closure_phase.h"
#include "model_closure_amplitude.h"
#include <vector>
#include <complex>

namespace Themis {

/*!
  \brief Defines a concentrid ring model based on the
  model_visibility_amplitude, model_closure_phase, and model_closure_amplitude
  classes.
  \details The concentric ring image is defined by the radius of the external 
  disc, the radius of the innner disc, and an overall flux normalization. 
  Note that this is not derived from 
  model_image because there is no meaning for the position angle.
  Parameter list:\n
  - parameters[0]: Total flux in Jy.
  - parameters[1]: Overall (outer) radius of the ring.
  - parameters[2]: \f$ \psi \f$ = The relative thickness.
*/
class model_ring : public model_visibility_amplitude, public model_closure_phase, public model_closure_amplitude
{
  public:
    
    model_ring();
    virtual ~model_ring() {};
    
    //! User-supplied function that returns the number of the parameters
    //! the model expects
    virtual inline size_t size() const { return 3; };
    
    //! A one-time generate function that permits model construction prior 
    //! to calling the visibility_amplitude, closure_phase, etc. for each 
    //! datum.  Takes a vector of parameters.
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
    comparison value. */
    virtual double closure_phase(datum_closure_phase& d, double acc);
      
    /*! Returns closure amplitude computed from the image given a 
    datum_closure_phase object, containing all of the accoutrements.  While 
    this provides access to the actual data value, the two could be separated 
    if necessary.  Also takes an accuracy parameter with the same units as 
    the data, indicating the accuracy with which the model must generate a 
    comparison value.  Note that this can be redefined in child classes. */
    virtual double closure_amplitude(datum_closure_amplitude& d, double acc);
    
    
  private:
    
    //! \brief Calculates the Bessel Function J1(x)
    double BesselJ1(double x);
    
    //! \brief Calculates the complex visibility amplitude
    std::complex<double> complex_visibility(double u, double v);

    double _V0;    //!< Total flux of the ring.
    double _Rext;  //!< Radius of the larger (outer) disk
    double _Rint;  //!< Radius of the smaller (inner) disk
};

};
#endif
