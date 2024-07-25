/*!
  \file model_shearing_spot.h
  \author Paul Tiede
  \date October 1, 2017
  \brief Header file for shearing spot class
  \details FILL IN LATER
*/
#ifndef VRT2_ED_SHEARING_SPOT_H
#define VRT2_ED_SHEARING_SPOT_H

#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>

#include "metric.h"
#include "fourvector.h"
#include "afv_keplerian.h"
#include "electron_density.h"

#include "vrt2_globs.h"
#include "vrt2_constants.h"
namespace VRT2 {
/* \brief Defines a shearing spot model for spot that is initial a spherical Gaussian spot, when measured in the proper time frame. This model is based on the VRT2 ED_SphericalOutflowingSpot model.

   \details FILL IN

   \warning NOT WORKING YET
 */

class ED_shearing_spot : public ElectronDensity
{
 public:
  ED_shearing_spot( Metric& g, double density_scale, double rSpot, double rISCO, AFV_Keplerian& afv, double t0, double r0, double theta0, double phi0);
  virtual ~ED_shearing_spot() {};

  virtual double get_density(const double t, const double r, const double theta, const double phi);


 protected:
  Metric& _g;
  const double _density_scale;
  const double _rspot;
  const double _rISCO;

  AFV_Keplerian& _afv; //Accretion flow velocity field (MUST BE KEPLERIAN)
  FourVector<double> _uCenter0;
  std::valarray<double> _xspot_center0;

  double delta_r(const double t, const double r, const double theta, const double phi);
};




}

#endif
