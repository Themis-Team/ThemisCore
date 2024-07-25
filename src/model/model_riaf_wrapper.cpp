/*!
  \file model_riaf_wrapper.cpp
  \author Mansour Karami
  \date  April, 2017
  \brief Implements extended RIAF image+flux model wrapper class.
  \details To be added
*/

#include "model_riaf_wrapper.h"

namespace Themis {

model_riaf_wrapper::model_riaf_wrapper(std::vector<double> frequencies, double M, double D)
  :model_riaf(frequencies, M, D)
{
  _parameters.resize(15);
}


model_riaf_wrapper::~model_riaf_wrapper()
{
 
}


void model_riaf_wrapper::generate_model(std::vector<double> parameters)
{
  //  ****  THERMAL/NON-THERMAL NORMALIZATIONS  ****  //
  /*
  // BH spin parameters
  _parameters[0] = parameters[0];  // Black hole spin parameter (-1 to 1).
  _parameters[1] = parameters[1];  // Cosine of the spin inclination relative to line of sight.
  
  // "Thermal" electron population density ...
  _parameters[2] = parameters[2];  // Normalization in \f${\rm cm}^{-3}\f$.
  _parameters[3] = -1.1;           // Radial power law.
  _parameters[4] = 1.0;            // h/r
  
  // "Thermal" electron population temperature ...
  _parameters[5] = parameters[3];  // Normalization in K.
  _parameters[6] = -0.84;          // Radial power law.
  
  // "Nonthermal" electron population density ...
  _parameters[7] = parameters[4];  // Normalization in \f${\rm cm}^{-3}\f$.
  _parameters[8] = -2.02;          // Radial power law.
  _parameters[9] = 1.0;            // h/r.
  
  _parameters[10] = 1.25;          // Spectral index.
  _parameters[11] = 100.0;         // Minimum Lorentz factor.
  _parameters[12] = 10.0;          // Plasma beta.
  _parameters[13] = 1.0;           // Sub-keplerian parameter
  _parameters[14] = parameters[5]; // Position angle (in model_image) in radians.
  */

  
  //  ****  DISK STRUCTURE  ****  //
  
  // BH spin parameters
  _parameters[0] = parameters[0];  // Black hole spin parameter (-1 to 1).
  _parameters[1] = parameters[1];  // Cosine of the spin inclination relative to line of sight.
  
  // "Thermal" electron population density ...
  _parameters[2] = parameters[2];  // Normalization in \f${\rm cm}^{-3}\f$.
  _parameters[3] = -1.1;           // Radial power law.
  _parameters[4] = parameters[3];  // h/r
  
  // "Thermal" electron population temperature ...
  _parameters[5] = parameters[4];  // Normalization in K.
  _parameters[6] = -0.84;          // Radial power law.
  
  // "Nonthermal" electron population density ...
  _parameters[7] = parameters[5];  // Normalization in \f${\rm cm}^{-3}\f$.
  _parameters[8] = -2.02;          // Radial power law.
  _parameters[9] = _parameters[4]; // h/r.
  
  _parameters[10] = 1.25;          // Spectral index.
  _parameters[11] = 100.0;         // Minimum Lorentz factor.
  _parameters[12] = 10.0;          // Plasma beta.
  _parameters[13] = parameters[6]; // Sub-keplerian parameter  
  _parameters[14] = parameters[7]; // Position angle (in model_image) in radians.
  

  _generated_model = true;


  /*std::cout << " Spin "   << _parameters[0] <<
    " cos(inc) "            << _parameters[1] <<
    " e norm "              << _parameters[2] <<
    " e power-law "         << _parameters[3] <<
    " e h/r "               << _parameters[4] <<
    " T norm "              << _parameters[5] <<
    " T power-law "         << _parameters[6] <<
    " nte norm "            << _parameters[7] <<
    " nte power-law "       << _parameters[8] <<
    " nte h/r "             << _parameters[9] <<
    " nte spectral-index "  << _parameters[10] <<
    " nte min gamma "       << _parameters[11] <<
    " beta "                << _parameters[12] <<
    " sub-keplerian param " << _parameters[13] <<
    " position angle "      << _parameters[14] << std::endl;*/
        
}

};

