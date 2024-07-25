/*!
  \file model_image_score_sgra.h
  \author Hung-Yi Pu, Paul Tiede
  \date  Oct, 2021
  \brief Header file for score model class.
  \details 
*/

#ifndef THEMIS_MODEL_IMAGE_SCORE_SGRA_H_
#define THEMIS_MODEL_IMAGE_SCORE_SGRA_H_

#include "model_image.h"
#include "vrt2.h"
#include <string>
#include <vector>
#include <complex>

#include <mpi.h>

#ifndef VERBOSITY
#define VERBOSITY (0)
#endif

//    Parameter list:\n
//	- parameters[1] ... (M/R)/ (M0/R0)
//      - parameters[2] ...  Postion Angle

namespace Themis {

  
  class model_image_score_sgra : public model_image
  {
    
  public:
    
    //model_image_score_sgra(int Nray, double M, double D, double fov, double frequency);
    // Takes file names
    model_image_score_sgra(std::string image_file_name, std::string README_file_name, bool reflect_image=false, int window_function=0);
    //model_image_score_sgra(std::string image_file_name, double fovx, double fovy, double frequency=230e9);
    
    virtual ~model_image_score_sgra() {};
    
    //! A user-supplied function that returns the number of the parameters the model expects
    virtual inline size_t size() const { return 2; };

    //! Return a string that contains a unique identifying tag for use with the ThemisPy plotting features.
    virtual std::string model_tag() const;

    //! A one-time generate function that permits model construction prior to calling the visibility_amplitude, closure_phase, etc. for each datum.  Takes a vector of parameters.
    virtual void generate_model(std::vector<double> parameters);
    
    virtual void set_mpi_communicator(MPI_Comm comm) 
    {
      //std::cout << "model_image_sed_fitted_riaf proc " << MPI::COMM_WORLD.Get_rank() << " set comm" << std::endl;
      _comm=comm;
    };

    //! Returns complex visibility in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual std::complex<double> visibility(datum_visibility& d, double accuracy);
    
    //! Returns visibility ampitudes in Jy computed from the image given a datum_visibility_amplitude object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double visibility_amplitude(datum_visibility_amplitude& d, double accuracy);
    
    //! Returns closure phase in degrees computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double closure_phase(datum_closure_phase& d, double accuracy);
    
    //! Returns closure amplitude computed from the image given a datum_closure_phase object, containing all of the accoutrements.  While this provides access to the actual data value, the two could be separated if necessary.  Also takes an accuracy parameter with the same units as the data, indicating the accuracy with which the model must generate a comparison value.  Note that this can be redefined in child classes.
    virtual double closure_amplitude(datum_closure_amplitude& d, double accuracy);
      
  protected:
    MPI_Comm _comm;
    
    
  private:
    //==image information	
    //!!!!note that the image file is specified in model_image_score_sgra.cpp!!
    double _MoD; // M/D in uas
    double _mod; // M/D ratio, total flux in Jy       
    double _frequency; // obs frequncy (GHz) 

    std::string _image_file_name, _README_file_name;
    bool _reflect_image;
   
  public:
    virtual void generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, 
				std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta);   
  };
};
#endif
