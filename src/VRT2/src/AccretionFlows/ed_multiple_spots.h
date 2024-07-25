/*!
  \file ed_multiple_spots.h
  \author Paul Tiede
  \date October 1, 2017
  \brief Calculates density profile for multiple shearing spots.
  \details Calculates the density profile for an multiple initially gaussian spot falling into a kerr black hole. Our velocity vector field is given by a combination of keplerian and free-fall velocity field. Does this by integrating one curve for the spot and the uses the azimuthal and time symmetry of the Kerr metric to find all the curves. To calculate densiy we compute the density expansion factor in a table in the constructor of the class. We the perform a linear intepolation to find the correct values.
*/
#ifndef VRT2_ED_MULTIPLE_SPOTS_H
#define VRT2_ED_MULTIPLE_SPOTS_H

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

class ED_multiple_spots : public ElectronDensity
{
 public:
  /*
    Defines inalling spot class of initially gaussian spot
  */
  //spot_parameters:
  //  [0]: density_scale
  //  [1]: rspot
  //  [2]: t0
  //  [3]: r0
  //  [4]: theta0
  //  [5]: phi0

	//Creates shearing spot electron density where spot table is generated in the constructor
  ED_multiple_spots( Metric& g, double ri, 
                     AFV_ShearingInflow& afv, double infallRate, double subKep, 
                     double outer_radius,
                     std::vector<std::vector<double> > spot_parameters);

  virtual ~ED_multiple_spots() {};

  //Computes density of spot
  virtual double get_density(const double t, const double r, const double theta, const double phi);

  //Allows us to step through time by changing table zero
  inline void set_toffset(double toffset) {_toffset = toffset; };

  //Generate integral curve path
  void generate_table();
  //Import integral curve path
  void set_table(std::vector<std::vector<double> > spot_tables);
  //Output integral curve path
  void export_table(std::vector<std::vector<double> >& spot_tables);

 protected:
  Metric& _g;
  const double _ri; //switching radius to ballistic trajectories

  //Holds the initial spot parameters
  std::vector<std::vector<double> > _spot_parameters;

  AFV_ShearingInflow& _afv; //Accretion flow velocity field
  const double _infallRate, _subKep; //Parameters of accretion flow velocity field.
  const double _outer_radius; //outer radius to which to integrate the table to
  bool _table_gen;

  //Bookeeeping some arrays for speed, i.e. don't have to constantly allocate.
  std::valarray<double> _xinitial, _xnow;
  

  //Stores offset time for stepping through time of table
  double _toffset;


  //Spot center location tables
  std::vector<std::vector<double> > _xspot_tables;
  std::vector<std::vector<double> > _xspot_tablesRev; //Order reversed
  std::vector<double> _rArray, _tArray; //Arrays for interpolating
  void generate_spot_position_table();

  //Gets spot position in time and integral expansion factor and stores it in _xinitial
  void integral_curve(const double t, std::valarray<double>& xinitial);
  //Gets position ar radius and integral factor and stores it in _xnow
  void curve_now(const double r, std::valarray<double>& xnow);
 
  //Reparameterizes the vector field to eliminate singular behavior near the horizon
  double reparametrize(const double []);


  //Compute density for infalling and keplerian orbit.
  double infall_density(const double t, const double r, const double theta, const double phi);

  //Finds difference needed for the initial spot density function.
  inline double delta_r(const std::valarray<double>& x, const std::valarray<double>& xcenter);

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
inline void ED_multiple_spots::get_yscal(double h, double x, const double y[], const double dydx[], double yscal[]) const
{
  yscal[0] = std::fabs(y[0]) + std::fabs(dydx[0]*h) + TINY; //t
  yscal[1] = std::fabs(y[1]) + std::fabs(dydx[1]*h) + TINY; //r
  yscal[2] = VRT2_Constants::pi; //theta
  yscal[3] = std::fabs(y[3]) + std::fabs(dydx[3]*h) + TINY; //phi

}
#undef TINY



}

#endif
