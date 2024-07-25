/********************************************************************/
/***                                                              ***/
/***  Computes the scaling constants associated with the disk     ***/
/***  model presented in:                                         ***/
/***       Brodererick, A.E. & Loeb, A., 2006, ApJL, 636, 109     ***/
/***  based upon fits to best-fits for each a,THETA of the the    ***/
/***  spectrum.                                                   ***/
/***                                                              ***/
/***  The power laws are taken from Yuan, Quataert, Narayan, 2003.***/
/***                                                              ***/
/***  Takes a in [0,1] and THETA in degrees                       ***/
/***                                                              ***/
/********************************************************************/

#ifndef VRT2_SGRA_DISK_MODEL_PARAMETERS_2010_H
#define VRT2_SGRA_DISK_MODEL_PARAMETERS_2010_H

#include <fstream>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <iostream>
using namespace std;
#include <vector>
#include <algorithm>

//make mpi-able
#ifdef MPI_MAP
#include <mpi.h>
#endif


///////////////// POLYNOMIAL INTERPOLATION /////////////////////////////
namespace VRT2 {
class SgrA_PolintDiskModelParameters2010
{
 public:
  SgrA_PolintDiskModelParameters2010(std::string fname, int aorder, int thetaorder, double a=0, double THETA=0);

  void reset(double a, double THETA);
  void set_orders(int aorder, int thetaorder);

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
  double _a, _THETA;
  int _aorder,_thetaorder;

  std::vector<double> a_, theta_, ne_, Te_, nnth_; // Tables to interpolate on
  double amin_, amax_, thetamin_, thetamax_;
  int Na_, Ntheta_;

  double interpolate2D(const std::vector<double>& y, int na, int nth, double a, double theta) const; // Wrapper to do 2D polint stuff
  void polint(double xa[], double ya[], int n, double x, double &y, double &dy) const; // Polynomial interp from NR
};

};
#endif
