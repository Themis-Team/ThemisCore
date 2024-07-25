/*******************************************/
/****** Header file for dispersionrelation.cpp *****/
/* NOTES:
   By default is set to Vaccuum light propagation
   Needs Metric metric defined as global for dD_dx
   Has a single integer parameter for the dispersion relation
     corresponding to the mode number.
*/
/*******************************************/

// Only include once
#ifndef VRT2_DISPERSIONRELATION_H
#define VRT2_DISPERSIONRELATION_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;

// Local Headers
#include "metric.h"
#include "fourvector.h"
#include "vrt2_globs.h"

namespace VRT2 {
class DispersionRelation{

 public:
  // Constructors and Destructor
  // Don't initialize
  DispersionRelation(Metric& metric);
  // Initialize via y[]=(x_con[],k_cov[],other)
  DispersionRelation(double [], Metric& metric);
  // Initialize via x,k as 4-vecs
  DispersionRelation(FourVector<double>&, FourVector<double>&, Metric& metric);
  virtual ~DispersionRelation();

  // reinitialization
  virtual void reinitialize(double []);
  virtual void reinitialize(FourVector<double>&, FourVector<double>&);

  // Common functions
  virtual void get_fcns();

  // Functions:
  // _x and _k have already been defined
  // DispersionRelation Relation (reduces to k^2 in vacuum!)
  virtual double D(int);
  // Dispersion relation for zeroing (maybe nice factors)
  virtual double zeroing_D(int);

  // Derivatives of DispersionRelation Relation wrt k_cov 
  //  (initialized contravariant)
  //  _x and _k have  already been defined
  virtual FourVector<double>& dD_dk(int); 

  // Derivatives of DispersionRelation Relation wrt x_con 
  //  (initialized covariant)
  //  _x and _k have already been defined
  virtual FourVector<double>& dD_dx(int);

  // Returns the k which zeros D at a given x
  virtual FourVector<double>& Zero_D(int);

  // Check for defined quantities (|2=D,|3=dD_dk,|5=dD_dx)
  unsigned int defined;

  // Internal x, k, and return from Zero_D(kz)
  FourVector<double> _x, _k, _kz;

 protected:
  Metric& _g;

  // Internal D
  double _D;
  // Internal Coherence Length
  double _cl;
  // Internal dD_dk and dD_dx
  FourVector<double> _dD_dk, _dD_dx;
  // Utility function for zeroing the Dispersion Relation
  //  (pointer implementation ensures polymorphism
  DispersionRelation *_zdisp_ptr;
  FourVector<double>& Zero_D_wptr(int);

 private:
  // Utility functions for zeroing the Dispersion Relation
  int _modez;
  double Zero_D_Func(double);
  int zbrac_AEB(double&,double&,int=0);
  double rtflsp_AEB(double,double,double);
  double zriddr_AEB(double,double,double);
};

};
#endif

