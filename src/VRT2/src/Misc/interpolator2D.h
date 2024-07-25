/*********************************************************************************/
/*                                                                               */
/* General Interpolator class.                                                   */
/*  Expects that x and y are monotonically increasing and that all interpolating */
/*  tables are in iy + y.size()*ix order.                                        */
/*                                                                               */
/*********************************************************************************/


#ifndef VRT2_INTERPOLATOR2D_H
#define VRT2_INTERPOLATOR2D_H

#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>
#include <iostream>
#include <iomanip>

namespace VRT2 {
class Interpolator2D
{
 public:
  Interpolator2D();
  Interpolator2D(std::valarray<double> x, std::valarray<double> y, std::valarray<double> f);
  Interpolator2D(std::valarray<double> x, std::valarray<double> y, std::valarray<double> f,
		 std::valarray<double> dfdx, std::valarray<double> dfdy, std::valarray<double> d2fdxdy);

  // Setting after the fact
  void set_f(std::valarray<double>& x, std::valarray<double>& y, std::valarray<double>& f);
  void set_df(std::valarray<double>& dfdx, std::valarray<double>& dfdy, std::valarray<double>& d2fdxdy);
  void use_forward_difference() { fd_derivatives(); };

  // Element access
  inline std::valarray<double>& x();
  inline std::valarray<double>& y();
  inline std::valarray<double>& f();
  inline std::valarray<double>& dfdx();
  inline std::valarray<double>& dfdy();
  inline std::valarray<double>& d2fdxdy();

  // Limits access
  inline double xmax();
  inline double ymax();
  inline double xmin();
  inline double ymin();

  // Interpolation cell limits
  inline void cell_limits(double x, double y,
			  double& xlo, double& xhi,
			  double& ylo, double& yhi);

  // Different interpolation methods
  void linear_triangle(double x, double y, double& f);
  void linear_loglog(double x, double y, double& f);
  void linear_wdbndry(double x, double y, double& f);  
  void linear(double x, double y, double& f);
  void bicubic(double x, double y, double& f);

 private:
  // Tables
  std::valarray<double> _x, _y, _f;
  std::valarray<double> _dfdx, _dfdy, _d2fdxdy;

  // Check existence and/or make by finite
  //   differencing of the derivatives if necessary
  bool _derivatives_defined;
  void fd_derivatives();

  // Make sure that tables are right size
  void check_tables() const;

  // Index
  inline int index(int ix, int iy) const;

  // Find the interpolation cell
  // 3    2
  // ^ y
  // | x
  // 0 -> 1
  inline void get_cell(double x, double y, int ix[], int iy[]);

  // subroutines for bicubic interpolation
  void bcuint(double y[], double y1[], double y2[], double y12[], double x1l,
	      double x1u, double x2l, double x2u, double x1, double x2,
	      double &ansy, double &ansy1, double &ansy2);
  void bcucof(double y[], double y1[], double y2[], double y12[],
	      double d1, double d2, double c[5][5]);
};

// Element access
std::valarray<double>& Interpolator2D::x()
{
  return _x;
}
std::valarray<double>& Interpolator2D::y()
{
  return _y;
}
std::valarray<double>& Interpolator2D::f()
{
  return _f;
}
std::valarray<double>& Interpolator2D::dfdx()
{
  return _dfdx;
}
std::valarray<double>& Interpolator2D::dfdy()
{
  return _dfdy;
}
std::valarray<double>& Interpolator2D::d2fdxdy()
{
  return _d2fdxdy;
}

double Interpolator2D::xmax()
{
  return _x[_x.size()-1];
}
double Interpolator2D::ymax()
{
  return _y[_y.size()-1];
}
double Interpolator2D::xmin()
{
  return _x[0];
}
double Interpolator2D::ymin()
{
  return _y[0];
}

// Interpolation Cell limits
void Interpolator2D::cell_limits(double x, double y, 
				 double& xlo, double& xhi,
				 double& ylo, double& yhi)
{
  static int ix[4], iy[4];
  get_cell(x,y,ix,iy);

  xlo = _x[ix[0]];
  xhi = _x[ix[2]];
  ylo = _y[iy[0]];
  yhi = _y[iy[2]];
}


void Interpolator2D::get_cell(double x, double y, int ix[], int iy[])
{
  // The interpolation cell corners are labeled in the following manner
  //                   3    2
  //                   ^ y
  //                   | x
  //                   0 -> 1


  // Get first value greater than x or _x.end() if none is found
  double *iter = std::lower_bound(&_x[0], &_x[_x.size()], x);
  if (iter == &_x[_x.size()])  // If at end, make both limits the end
    for (int i=0; i<4; ++i)
      ix[i] = _x.size()-1;
  else if (iter == &_x[0]) // If at begin, make both limits the begin
    for (int i=0; i<4; ++i)
      ix[i] = 0;
  else { // okay
    ix[1] = ix[2] = iter - &_x[0];
    ix[0] = ix[3] = ix[1] - 1;
  }

  // Get first value greater than y or _y.end() if none is found
  iter = std::lower_bound(&_y[0], &_y[_y.size()], y);
  if (iter == &_y[_y.size()])  // If at end, make both limits the end
    for (int i=0; i<4; ++i)
      iy[i] = _y.size()-1;
  else if (iter == &_y[0]) // If at begin, make both limits the begin
    for (int i=0; i<4; ++i)
      iy[i] = 0;
  else { // okay
    iy[2] = iy[3] = iter - &_y[0];
    iy[0] = iy[1] = iy[2] - 1;
  }
}

};
#endif
