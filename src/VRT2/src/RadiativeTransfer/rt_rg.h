/***************************************************/
/****** Header file for radiativetransfer.cpp ******/
/* Notes:                                          */
/*   Describes interface for RT.                   */
/***************************************************/

// Only include once
#ifndef VRT2_RT_RUNGE_KUTTA_H
#define VRT2_RT_RUNGE_KUTTA_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <vector>
#include <valarray>
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <fstream>

// Special Headers
#include "metric.h"
#include "fourvector.h"
#include "radiativetransfer.h"
#include "vrt2_globs.h"

namespace VRT2 {
class RT_RungeKutta : public RadiativeTransfer
{
 public:
  // Constructor
  RT_RungeKutta(Metric& g);
  RT_RungeKutta(const double y[], Metric& g);
  RT_RungeKutta(FourVector<double>& x,FourVector<double>& k, Metric& g);
  virtual ~RT_RungeKutta() { };

  // Integrated Stokes parameters
  virtual std::valarray<double>& IQUV_integrate(std::vector<double> ya[],std::vector<double> dydxa[],std::valarray<double>& iquv0);

  // Stable step size limiter (returns the minimum of the current step or the stable step)
  virtual double stable_step_size(double h, const double y[], const double dydx[]);


  //private:
 protected:
  // Local pointers to ray positions
  std::vector<double> *_ya, *_dydxa;

  // Linear interpolation to get x,k and dx,dk
  inline void interp(double lambda, double y[], double dydx[]);

  // Numerics for integration
  void derivs(double,const double[],double[]);
  int rkqs(double[],double[],int,double&,double,double,double[],double&,double&);
  void rkck(double [],double [],int,double,double,double [],double []);


  virtual void dump_ray(std::string) {};


  bool _output_flag;
};






///////////////////////// INLINED FUNCTIONS ////////////////////////////////////////////

// Inerpolate to a given lambda
void RT_RungeKutta::interp(double lambda, double y[], double dydx[])
{

  if (lambda<=_ya[0][0]){
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int j=0; j<8; ++j){
      y[j] = _ya[j+1][0];
      dydx[j] = _dydxa[j+1][0];
    }
  }
  else if (lambda>=_ya[0][_ya[0].size()-1]){
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int j=0; j<8; ++j){
      y[j] = _ya[j+1][_ya[0].size()-1];
      dydx[j] = _dydxa[j+1][_ya[0].size()-1];
    }
  }
  else{
    double* iter = std::lower_bound(&_ya[0][0],&_ya[0][_ya[0].size()],lambda);
    int i = iter - &_ya[0][0];
    double dx = (lambda - _ya[0][i-1])/(_ya[0][i]-_ya[0][i-1]);
    double mdx = 1.0-dx;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int j=0; j<8; ++j){
      y[j] = dx*_ya[j+1][i] + mdx*_ya[j+1][i-1];
      dydx[j] = dx*_dydxa[j+1][i] + mdx*_dydxa[j+1][i-1];
    }
  }
}

};
#endif
