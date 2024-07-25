/*!
  \file ed_shearing_infall_spot.h
  \author Paul Tiede
  \date October 1, 2017
  \brief Calculates density profile for a shearing infalling spot.
  \details Calculates the density profile for an initially gaussian spot falling into a kerr black hole. Our velocity vector field is given by a combination of keplerian and free-fall velocity field. Does this by integrating one curve for the spot and the uses the azimuthal and time symmetry of the Kerr metric to find all the curves. To calulcate densiy we compute the density expansion factor in a table in the constructor of the class. We the perform a linear intepolation to find the correct values.
*/
#ifndef VRT2_ED_SHEARING_INFALL_SPOT_H
#define VRT2_ED_SHEARING_INFALL_SPOT_H

#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "metric.h"
#include "fourvector.h"
#include "afv_shearing_inflow.h"
#include "electron_density.h"

#include "fast_math.h"
#include "interpolator1D.h"

#include "vrt2_globs.h"
#include "vrt2_constants.h"
namespace VRT2 {
/* \brief Defines a shearing spot model for spot that is initial a spherical Gaussian spot, when measured in the proper time frame.

   \details Electron density class for a shearing infalling spot. Assumes the accretion velocity field is the shearing infalling spot field.

   \warning 
 */

class ED_shearing_infall_spot : public ElectronDensity
{
 public:
  /*
    Defines inalling spot class of initially gaussian spot
  */

	//Creates shearing spot electron density where spot table is generated in the constructor
  ED_shearing_infall_spot( Metric& g, double density_scale, double rSpot, double ri, AFV_ShearingInflow& afv, double infallRate, double subKep, double outer_radius, double t0, double r0, double theta0, double phi0);

  virtual ~ED_shearing_infall_spot() {};

  //Computes density of spot
  virtual double get_density(const double t, const double r, const double theta, const double phi);

  //Allows us to step through time by changing table zero
  inline void set_toffset(double toffset) {_toffset = toffset; };

  //Generate integral curve path
  void generate_table();
  //Import integral curve path
  void set_table(std::vector<double> spot_tables[4]);
  //Output integral curve path
  void export_table(std::vector<double> spot_tables[4]);

 protected:
  Metric& _g;
  const double _density_scale; //Sets density scale of the spot
  const double _rspot; //Width of spot (standard deviation of the Gaussian spot profile.
  const double _ri; //Initial radius of spot center

  AFV_ShearingInflow& _afv; //Accretion flow velocity field
  const double _infallRate, _subKep; //Parameters of accretion flow velocity field.
  const double _outer_radius; //outer radius to which to integrate the table to
  bool _table_gen;

  //Stores interppolated value of location of spot center
  std::valarray<double> _xCenter, _xN;
  
  std::valarray<double> _xspot_center0;


  //Stores offset time for stepping through time of table
  double _toffset;


  //Spot center location tables
  std::vector<double> _xspot_tables[4];
  std::vector<double> _xspot_tablesRev[4]; //Order reversed
  std::vector<double> _rArray, _tArray; //Arrays for interpolating
  void generate_spot_position_table();

  //Gets spot position in time and integral expansion factor
  std::valarray<double>& integral_curve(const double t);
  //Gets position ar radius and integral factor
  std::valarray<double>& curve_now(const double r);
 
  //Reparameterizes the vector field to eliminate singular behavior near the horizon
  double reparametrize(const double []);


  //Compute density for infalling and keplerian orbit.
  double infall_density(const double t, const double r, const double theta, const double phi);

  //Finds difference needed for the initial spot density function.
  double delta_r(const double t, const double r, const double theta, const double phi);


  //reparameterization stuff for near horizon issues
  //double reparameterization(const double []);
  //Integrator functions
  void derivs(double, const double[], double[]);
  inline void get_yscal(double, double, const double [], const double [], double []) const;

  
  // Numerics
  int rkqs(double[],double[],int,double&,double,double,double[],double&,double&);
  void rkck(double [],double [],int,double,double,double [],double []);  
};




/*** Relative error scalings ***/
#define TINY 1.0e-2; //Very sensitive
inline void ED_shearing_infall_spot::get_yscal(double h, double x, const double y[], const double dydx[], double yscal[]) const
{
  yscal[0] = std::fabs(y[0]) + std::fabs(dydx[0]*h) + TINY; //t
  yscal[1] = std::fabs(y[1]) + std::fabs(dydx[1]*h) + TINY; //r
  yscal[2] = VRT2_Constants::pi; //theta
  yscal[3] = std::fabs(y[3]) + std::fabs(dydx[3]*h) + TINY; //phi

}
#undef TINY



}

#endif
