/********************************************************************/
/***                                                              ***/
/***  Computes the scaling constants associated with the jet      ***/
/***  model presented in:                                         ***/
/***       Brodererick, A.E. & Loeb, A., 2009, ApJ, 697, 1164     ***/
/***  based upon fits to best-fits for each a, loading radius,    ***/
/***  and IR/Optical flux correction factor to the spectrum.      ***/
/***                                                              ***/
/***                                                              ***/
/***  Takes a in [0,1], rload in in degrees                       ***/
/***                                                              ***/
/********************************************************************/

#ifndef VRT2_M87_JET_MODEL_PARAMETERS_2012_H
#define VRT2_M87_JET_MODEL_PARAMETERS_2012_H

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include "kerr.h"

namespace VRT2 {
///////////////// POLYNOMIAL INTERPOLATION /////////////////////////////
class M87_PolintJetModelParameters2012
{
 public:
  M87_PolintJetModelParameters2012(std::string fname, int ccforder, int aorder, int rlorder, double ccf=0, double a=0, double rl=0);
  ~M87_PolintJetModelParameters2012();

  void reset(double ccf, double a, double rl);
  void set_orders(int ccforder, int aorder, int rlorder);

  // Non-thermal electron injection parameters
  double nj_norm() const;
  double nj_index() const;
  double nj_gammamin() const { return 1.0e2; };

  // Magnetic field normalization
  double bj_norm() const;

  // Theta
  double THETA() const { return 25.0; };

  // GammaMax
  double GammaMax() const { return 5.0; };

  // Jet structure parameters
  double p() const { return 2.0/3.0; };
  double disk_inner_edge_radius() const;
  double opening_angle() const { return 10.0; }
  

 private:
  double _c, _a, _r;
  int _corder, _aorder, _rorder;


  double cmin_, cmax_, amin_, amax_, *rmin_, *rmax_;
  double *c_, *a_, **r_;
  double ***nj_, ***bj_, ***aj_;
  int Nc_, Na_, *Nr_;


  double interpolate2D(double **fg, const double *xg, const double *yg, int Nxg, int Nyg, int npx, int npy, double x, double y) const;

  double interpolate3D(double ***fg, int npc, int npa, int npr, double c, double a, double r) const;

  void polint(double xa[], double ya[], int n, double x, double &y, double &dy) const; // Polynomial interp from NR

};

};
#endif


