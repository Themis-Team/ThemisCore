/*************************************************/
/****** Header file for nullgeodesicc.cpp ******/
/* Notes:
   Traces null geodesics.  Uses dispersionrelation.cpp
*/
/*************************************************/

// Only include once
#ifndef VRT2_NULLGEODESIC_H
#define VRT2_NULLGEODESIC_H

// Standard Library Header
#include <stdio.h>
#include <vector>
#include <valarray>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <math.h>
using namespace std;
#include <algorithm>

// Special Headers
#include "metric.h"
#include "fourvector.h"
#include "ray.h"
#include "vrt2_constants.h"
#include "dispersionrelation.h"
#include "radiativetransfer.h"
#include "stopcondition.h"
#include "vrt2_globs.h"

#define NDIM_NAR 10 // Number of differential equations

namespace VRT2 {
class NullGeodesic : public Ray
{
 public:
  NullGeodesic(Metric& metric, RadiativeTransfer& rt, StopCondition& stop);

  NullGeodesic(double y[], Metric& metric, RadiativeTransfer& rt, StopCondition& stop);

  NullGeodesic(FourVector<double>& x0,FourVector<double>& k0,
		Metric& metric, RadiativeTransfer& rt, StopCondition& stop);

  virtual ~NullGeodesic() {};

  virtual void reinitialize(FourVector<double>&,FourVector<double>&);

  // Prpagation  
  virtual std::vector<std::string> propagate(double h, std::string output="!");

  // Functions for output
  virtual std::valarray<double> IQUV();
  virtual double tau();
  virtual double D();
  virtual void output_ray(std::string,int);
  virtual void output_ray(std::ostream&,int);

 private:
  Metric& _g;                               // Local Metric
  FourVector<double> _x0, _k0;              // Local x0 and k0
  DispersionRelation _disp;                 // Local disp
  RadiativeTransfer& _rt;                   // Local rad
  StopCondition& _stop;                     // Local stop
  double _y_save[NDIM_NAR];                 // Save step check
  std::vector<double> _y_ray[NDIM_NAR+1];   // Local ray coords (all points)
  std::vector<double> _dydx_ray[NDIM_NAR+1];// Local ray derivs (all points)
  double _tau_int;                          // tau integrals
  double _D_int;                            // D integrals
  std::valarray<double> _iquv;              // Stokes parameters

  // Functions for derivs
  void derivs(double,double[],double[]);
  inline double reparametrize(double[]);
  inline void get_yscal(double,double,double[],double[],double[]);

  // dl2 and dl2_max;
  inline double dl2(double[],double[],DispersionRelation&);
  inline double dl2_max(double[],double[],DispersionRelation&);

  // Functions for conditions on propagation
  int save_y(double y[], int set);
  void add_to_y_ray(double,double[],double[]);
  void backup_y_ray(double);

  // Numerics
  int rkqs(double[],double[],int,double&,double,double,double[],double&,double&);
  void rkck(double [],double [],int,double,double,double [],double []);
};
};
#endif
