/***************************************************/
/*** A Qualitative Force-Free Jet Model          ***/
/*                                                 */
/* This is based upon Tchekhovskoy et al. (2008).  */
/*                                                 */
/* What must be specified are:                     */
/*   (1) A stream function, psi                    */
/*   (2) Angular frequency as a func. psi          */
/*   (3) And the density foot print as a func. psi */
/*                                                 */
/* This encapsulates the computation of the        */
/* density, velocity and magnetic field geometry.  */
/*                                                 */
/***************************************************/

#ifndef VRT2_FORCE_FREE_JET_H
#define VRT2_FORCE_FREE_JET_H

#include <valarray>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "vrt2_constants.h"
#include "magnetic_field.h"
#include "electron_density.h"
#include "accretion_flow_velocity.h"
#include "metric.h"
#include "fourvector.h"
#include "vrt2_globs.h"

namespace VRT2 {

#define CONSERVATIVE_MASS_TRANSPORT // Use the continuity equation to determine the density

// Encapsulates everything
class ForceFreeJet
{
 public:
  ForceFreeJet(Metric& g, double p, double r_inner_edge, double r_foot_print, double r_load, double B0, double n0, double gamma_max=0.0);
  
  // Access to interfaces
  MagneticField& mf();
  AccretionFlowVelocity& afv();
  ElectronDensity& ed();



 private:

  Metric& _g;

  double _p;
  double _ri;
  double _rj;
  double _dtheta;
  double _rl;
  double _B0;
  double _n0;

  double _uF2;
  double _bF2;
  double _gamma;
  double _gamma_max;
  double _beta;

  ///////////////////////////
  //
  // Defines various structure function that define the jet
  //
  // Stream Function Details
  double psi(double r, double theta) const;
  double dpsi_dr(double r, double theta) const;
  double dpsi_dth(double r, double theta) const;
  // Omega as a function of stream function
  double _psii;  
  double OmegaZAMO(double r, double theta) const;
  double Omega(double psi) const;
  // Density function as a function of stream function
  double _psij;
  double F(double psi) const;

  void get_F(double r);
  void get_F2(double rh, double rj);
  std::vector<double> _Fv, _psv;


  //////////////////////////
  //
  // Internal stuff to compute the magnetic field, velocity and density
  //
  FourVector<double> _bF, _uF, _b, _u;
  double _n;
  void compute_ubn(double r, double theta);
  void compute_all(double r, double theta);
  FourVector<double>& magnetic_field(double r, double theta);
  FourVector<double>& velocity(double r, double theta);
  double density(double r, double theta);


  // Interfaces
  class MF_FFJ : public MagneticField
  {
   public:
    MF_FFJ(Metric& g, ForceFreeJet& ffj) : MagneticField(g), _ffj(ffj) {};
    virtual ~MF_FFJ() {};
    virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);
   private:
    ForceFreeJet& _ffj;
  };
  class AFV_FFJ : public AccretionFlowVelocity
  {
   public:
    AFV_FFJ(Metric& g, ForceFreeJet& ffj) : AccretionFlowVelocity(g), _ffj(ffj) {};
    virtual ~AFV_FFJ() {};
    virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);

   private:
    ForceFreeJet& _ffj;
  };
  class ED_FFJ : public ElectronDensity
  {
   public:
    ED_FFJ(ForceFreeJet& ffj) : _ffj(ffj) {};
    virtual ~ED_FFJ() {};
    virtual double get_density(double t, double r, double theta, double phi);
   private:
    ForceFreeJet& _ffj;
  };

  AFV_FFJ _afv;
  ED_FFJ _ed;
  MF_FFJ _mf;

  bool past_light_cylinder_proximity_limit();

};





};
#endif
