/********************************************************************/
/***                                                              ***/
/***  Computes the scaling constants associated with the disk     ***/
/***  model presented in:                                         ***/
/***       Brodererick, A.E. & Loeb, A., 2006, ApJL, 636, 109     ***/
/***  based upon fits to best-fits for each a,THETA,epsilon of    ***/
/***  the spectrum.                                               ***/
/***                                                              ***/
/***  The power laws are taken from Yuan, Quataert, Narayan, 2003.***/
/***                                                              ***/
/***  Assumes the tabulated values are on rectalinear a-theta     ***/
/***  grids with sizes that depend upon epsilon.                  ***/
/***                                                              ***/
/***                                                              ***/
/***                                                              ***/
/***                                                              ***/
/********************************************************************/

#ifndef VRT2_SGRA_DISK_MODEL_PARAMETERS_HT_2011_H
#define VRT2_SGRA_DISK_MODEL_PARAMETERS_HT_2011_H

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <math.h>
using namespace std;
#include <vector>
#include <algorithm>


///////////////// POLYNOMIAL INTERPOLATION /////////////////////////////
namespace VRT2 {
class SgrA_PolintDiskModelParametersHT2011
{
 public:
  SgrA_PolintDiskModelParametersHT2011(std::string fname, int aorder, int thetaorder, int eorder, double a=0, double THETA=0, double e=0);
  ~SgrA_PolintDiskModelParametersHT2011();

  void reset(double a, double THETA, double e);
  void set_orders(int aorder, int thetaorder, int eorder);

  // Thermal electron density
  double ne_norm() const;
  double ne_index() const { return -1.1; };
  double ne_height() const { return 1.0; };
  // Thermal electron temperature
  double Te_norm() const;
  double Te_index() const { return -0.84; };
  double Te_height() const { return 1.0; };
  // Non-thermal electron density
  double nnth_norm() const;
  double nnth_index() const { return -2.02; };
  double nnth_height() const { return 1.0; };

 private:
  double _a, _THETA, _e;
  int _aorder,_thetaorder,_eorder;


  double *amin_, *amax_, *tmin_, *tmax_, emin_, emax_;
  double **a_, **t_, *e_;
  double ***ne_, ***Te_, ***nnth_;
  int *Na_, *Nt_, Ne_;



  double interpolate2D(double **fg, const double *xg, const double *yg, int Nxg, int Nyg, int npx, int npy, double x, double y) const;

  double interpolate3D(double ***fg, int npa, int npt, int npe, double a, double t, double e) const;

  void polint(double xa[], double ya[], int n, double x, double &y, double &dy) const; // Polynomial interp from NR

  int index(int ia, int it, int ie) const;
};
};
#endif
