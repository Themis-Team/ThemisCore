/*!
  \file interpolator2D.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements a general purpose 2D interpolator class that implements a variety of interpolation schemes.
  \details To be added
*/

#include "interpolator2D.h"

#include <iostream>
#include <iomanip>

namespace Themis {

Interpolator2D::Interpolator2D()
  : _x(0), _y(0), _f(0),
    _dfdx(0), _dfdy(0), _d2fdxdy(0), _spline_derivs(0),
    _derivatives_defined(false)
{
}

Interpolator2D::Interpolator2D(std::valarray<double> x,
			       std::valarray<double> y,
			       std::valarray<double> f)
  : _x(x), _y(y), _f(f),
    _dfdx(0), _dfdy(0), _d2fdxdy(0),_spline_derivs(0),
    _derivatives_defined(false)
{
  check_tables();
}

Interpolator2D::Interpolator2D(std::valarray<double> x,
			       std::valarray<double> y,
			       std::valarray<double> f,
			       std::valarray<double> dfdx,
			       std::valarray<double> dfdy,
			       std::valarray<double> d2fdxdy)
  : _x(x), _y(y), _f(f),
    _dfdx(dfdx), _dfdy(dfdy), _d2fdxdy(d2fdxdy),_spline_derivs(0),
    _derivatives_defined(false)
{
  if (dfdx.size() && dfdy.size() && d2fdxdy.size())
  {
    gen_spline_derivs();
    _derivatives_defined = true;
  }
  check_tables();
}

// Setting after the fact
void Interpolator2D::set_f(std::valarray<double>& x, std::valarray<double>& y, std::valarray<double>& f)
{
  _x.resize(x.size());
  _y.resize(y.size());
  _f.resize(f.size());
  _x = x;
  _y = y;
  _f = f;

  check_tables();
}
void Interpolator2D::set_df(std::valarray<double>& dfdx, std::valarray<double>& dfdy, std::valarray<double>& d2fdxdy)
{
  _dfdx.resize(dfdx.size());
  _dfdy.resize(dfdy.size());
  _d2fdxdy.resize(d2fdxdy.size());

  _dfdx = dfdx;
  _dfdy = dfdy;
  _d2fdxdy = d2fdxdy;
  
  gen_spline_derivs();
  _derivatives_defined = true;

  check_tables();
}

void Interpolator2D::check_tables() const
{
  // Check that _f is the right size
  if (_x.size()*_y.size() != _f.size())
    std::cerr << "Inconsitent interpolation table (f)!\n";

  if (_derivatives_defined) {
    // Check that _dfdx is the right size
    if (_x.size()*_y.size() != _dfdx.size())
      std::cerr << "Inconsitent interpolation table (dfdx)!\n";
    // Check that _dfdy is the right size
    if (_x.size()*_y.size() != _dfdy.size())
      std::cerr << "Inconsitent interpolation table (dfdy)!\n";
    // Check that _d2fdxdy is the right size
    if (_x.size()*_y.size() != _d2fdxdy.size())
      std::cerr << "Inconsitent interpolation table (d2fdxdy)!\n";
    if (_x.size()*_y.size() != _spline_derivs.size())
      std::cerr << "Inconsistent interpolation table (spline_derivs)!\n";
  }
}

int Interpolator2D::index(int ix, int iy) const
{
  return ( iy + ix*_y.size() );
}

void Interpolator2D::fd_derivatives()
{
  _dfdx.resize(_x.size()*_y.size());
  _dfdy.resize(_x.size()*_y.size());
  _d2fdxdy.resize(_x.size()*_y.size());
  _spline_derivs.resize(_x.size()*_y.size());

  int ind;
  for (int ix=0; ix<int(_x.size()); ++ix)
    for (int iy=0; iy<int(_y.size()); ++iy) {
      ind = index(ix,iy);
      _dfdx[ind] = _dfdy[ind] = _d2fdxdy[ind] = _spline_derivs[ind] = 0.0;
    }

  for (int ix=1; ix<int(_x.size()-1); ++ix)
    for (int iy=0; iy<int(_y.size()); ++iy)
      _dfdx[index(ix,iy)] = (_f[index(ix+1,iy)] - _f[index(ix-1,iy)])/(_x[ix+1]-_x[ix-1]);

  for (int ix=0; ix<int(_x.size()); ++ix)
    for (int iy=1; iy<int(_y.size()-1); ++iy)
      _dfdy[index(ix,iy)] = (_f[index(ix,iy+1)] - _f[index(ix,iy-1)])/(_y[iy+1]-_y[iy-1]);

  // Could choose either way, but neither is better
  for (int ix=1; ix<int(_x.size()-1); ++ix)
    for (int iy=1; iy<int(_y.size()-1); ++iy)
      _d2fdxdy[index(ix,iy)] = (_dfdx[index(ix,iy+1)] - _dfdx[index(ix,iy-1)])/(_y[iy+1]-_y[iy-1]);
  
  gen_spline_derivs();
  // Now say that derivatives are defined
  _derivatives_defined = true;
}

void Interpolator2D::gen_spline_derivs()
{
  _spline_derivs.resize(_x.size()*_y.size());
  for ( int ix=0; ix < int(_x.size()); ++ix)
    for ( int iy=0; iy < int(_y.size()); ++iy)
      _spline_derivs[index(ix,iy)] = 0;

  //Now use spline interpolator to find derivs
  for ( int ix = 0; ix < int(_x.size()); ++ix)
  {
    //Descend down the rows of the fij matrix interpolating each row
    
    //Spline boundary conditions already set so move into tridiagonal
    int n = _y.size();
    std::valarray<double> u(0.0, n);
    for ( int iy = 1; iy < n-1; iy++)
    {
      double sig = (_y[iy]-_y[iy-1])/(_y[iy+1]-_y[iy-1]);
      double p = sig*_f[index(ix,iy-1)]+2.0;
      _spline_derivs[index(ix,iy)] = (sig-1.0)/p;
      u[iy] = (_f[index(ix,iy+1)]-_f[index(ix,iy)])/(_y[iy+1] - _y[iy])
            - (_f[index(ix,iy)] - _f[index(ix,iy-1)])/(_y[iy]-_y[iy-1]);
      u[iy] = ( 6.0*u[iy]/(_y[iy+1] - _y[iy-1]) -sig*u[iy-1])/p;
    }

    //Now do backsubstitution loop of tridiagonal algorithm
    for ( int iy = n-2; iy>=0; iy--)
      _spline_derivs[index(ix,iy)] = _spline_derivs[index(ix,iy)]*_spline_derivs[index(ix,iy+1)] + u[iy];
  }
}


/***********************  Specific Interpolation Schemes **********************/
/*** Split each four corner cell into two planar triangles ***/
void Interpolator2D::linear_triangle(double x, double y, double& f)
{
  static int ix[4], iy[4], ind[4];

  // Get cell corners
  get_cell(x,y,ix,iy);

  // Get indicies
  for (int i=0; i<4; ++i) {
    ind[i] = index(ix[i],iy[i]);
  }

  // Position of maximum density
  int imax=0;
  for (int i=1; i<4; ++i)
    if (_f[ind[i]]>_f[ind[imax]])
      imax = i;
  
  // Position of opposite corner
  int iopp=(imax+2)%4;

  // Position of corner at same y adjacent to imax
  int ix_adj = imax + 1 - (imax%2)*2;
  // Position of corner at same x adjacent to imax
  int iy_adj = ( imax + 3 - (imax%2)*2 )%4;

  // dx and dy from imax
  double dx, dy;
  if (ix[imax]!=ix[ix_adj])
    dx = x-_x[ix[imax]];
  else
    dx = 0.0;

  if (iy[imax]!=iy[iy_adj])
    dy = y-_y[iy[imax]];
  else
    dy = 0.0;

  // Step sizes
  double Dx = _x[ix[ix_adj]] - _x[ix[imax]]; // in x direction
  double Dy = _y[iy[iy_adj]] - _y[iy[imax]]; // in y direction

  // Choose triangle to interpolate in
  double fx, fy, f0;
  if ( std::fabs(dy) < (std::fabs(Dy) - std::fabs(Dy/Dx * dx)) ) { // on imax side
    f0 = _f[ind[imax]];
    fx = _f[ind[ix_adj]];
    fy = _f[ind[iy_adj]];
  }
  else { // on iopp side
    f0 = _f[ind[iopp]];
    fx = _f[ind[iy_adj]]; // Note that ix and iy adjacent switch now!
    fy = _f[ind[ix_adj]];

    Dx = -Dx;
    Dy = -Dy;
    dx += Dx;
    dy += Dy;
  }

  // Do interpolation
  double dfdx = (fx-f0)/Dx;
  double dfdy = (fy-f0)/Dy;
  f = dfdx*dx + dfdy*dy + f0;
}
/*** Bilinear interpolation in loglog coordinates ***/
void Interpolator2D::linear_loglog(double x, double y, double& f)
{
  static int ix[4], iy[4], ind[4];

  // Get cell corners
  get_cell(x,y,ix,iy);

  // Get indicies
  for (int i=0; i<4; ++i) {
    ind[i] = index(ix[i],iy[i]);
  }

  // If x & y !=0
  double lx, ly;
  double lf0, lf1, lf2, lf3;
  double lx_hi, lx_lo, ly_hi, ly_lo;
  if (_x[ix[0]] && _y[iy[0]]) {
    lx_hi = std::log(_x[ix[2]]);
    lx_lo = std::log(_x[ix[0]]);
    ly_hi = std::log(_y[iy[2]]);
    ly_lo = std::log(_y[iy[0]]);

    lx = std::log(x);
    ly = std::log(y);

    lf0 = std::log(_f[ind[0]]);
    lf1 = std::log(_f[ind[1]]);
    lf2 = std::log(_f[ind[2]]);
    lf3 = std::log(_f[ind[3]]);
  }
  else {
    lx_hi = _x[ix[2]];
    lx_lo = _x[ix[0]];
    ly_hi = _y[iy[2]];
    ly_lo = _y[iy[0]];

    lx = x;
    ly = y;

    lf0 = _f[ind[0]];
    lf1 = _f[ind[1]];
    lf2 = _f[ind[2]];
    lf3 = _f[ind[3]];
  }
    


  // 1.0/step sizes
  double oDx = 1.0/(lx_hi - lx_lo); // in x direction
  double oDy = 1.0/(ly_hi - ly_lo); // in y direction

  // dx and dy from imax
  double dx, dy;
  if (ix[0]!=ix[2])
    dx = (lx-lx_lo)*oDx;
  else
    dx = 0.0;

  if (iy[0]!=iy[2])
    dy = (ly-ly_lo)*oDy;
  else
    dy = 0.0;

  f = (1.0-dx)*(1.0-dy) * lf0
    + dx*(1.0-dy) * lf1
    + dx*dy * lf2
    + (1.0-dx)*dy * lf3;

  if (lx!=x && ly!=y)
    f = std::exp(f);
}

/*** Bilinear interpolation with extra derivative information at the boundaries ***/
void Interpolator2D::linear_wdbndry(double x, double y, double& f)
{
  // If derivatives aren't defined, using foward differences on the grid
  //  reduces to the normal bilinear interpolation
  if (!_derivatives_defined)
    linear(x,y,f);

  static int ix[4], iy[4], ind[4];

  // Get cell corners
  get_cell(x,y,ix,iy);

  // Get indicies
  for (int i=0; i<4; ++i) {
    ind[i] = index(ix[i],iy[i]);
  }

  // 1.0/step sizes
  double oDx = 1.0/(_x[ix[2]] - _x[ix[0]]); // in x direction
  double oDy = 1.0/(_y[iy[2]] - _y[iy[0]]); // in y direction

  // dx and dy
  double dx, dy;
  if (ix[0]!=ix[2])
    dx = (x-_x[ix[0]])*oDx;
  else
    dx = 0.0;

  if (iy[0]!=iy[2])
    dy = (y-_y[iy[0]])*oDy;
  else
    dy = 0.0;

  // Non-Boundary points -> just normal bilinear interpolation
  if ( _f[ind[0]] != 0.0 && _f[ind[1]] != 0.0 && _f[ind[2]] != 0.0 && _f[ind[3]] != 0.0 ) {
    f = (1.0-dx)*(1.0-dy) * _f[ind[0]]
      + dx*(1.0-dy) * _f[ind[1]]
      + dx*dy * _f[ind[2]]
      + (1.0-dx)*dy * _f[ind[3]];


    return;
  }
  else if (_f[ind[0]] != 0.0 || _f[ind[1]] != 0.0 || _f[ind[2]] != 0.0 || _f[ind[3]] != 0.0 ) {

    // Begin interpolation at boundaries
    //  Interpolation is performed in two stages:
    // (1) interpolate in x direction to get g_lo and g_hi
    // (2) interpolate in y direction on g to get f
    // At each step use the smaller (more quickly vanishing) of the
    //  finite difference or given derivative for points
    //  with vanishing pairs.

    double DfDy_lo, DfDy_hi, D2fDyDx_lo, D2fDyDx_hi;
    double DfDx_lo, DfDx_hi;
    double g_lo, g_hi, DgDy_lo, DgDy_hi, DgDy;
    
    // Get the DfDy's at the low y and high y
    DfDy_lo = oDy*(_f[ind[3]] - _f[ind[0]]);
    if (_f[ind[3]]==0.0)
      DfDy_lo = std::min( DfDy_lo , _dfdy[ind[0]] );
    else if (_f[ind[0]]==0.0)
      DfDy_lo = std::max( DfDy_lo , _dfdy[ind[3]] );

    DfDy_hi = oDy*(_f[ind[2]] - _f[ind[1]]);
    if (_f[ind[2]]==0.0)
      DfDy_hi = std::min( DfDy_lo , _dfdy[ind[1]] );
    else if (_f[ind[1]]==0.0)
      DfDy_hi = std::max( DfDy_lo , _dfdy[ind[2]] );

    // Get the DfDx's and D2fDyDx's
    DfDx_lo = oDx*(_f[ind[1]] - _f[ind[0]]);
    D2fDyDx_lo = oDx*(DfDy_hi - DfDy_lo);
    if (_f[ind[1]]==0.0) {
      DfDx_lo = std::min( DfDx_lo , _dfdx[ind[0]] );
      D2fDyDx_lo = std::min( D2fDyDx_lo, _d2fdxdy[ind[0]] );
    }
    else if (_f[ind[0]]==0.0) {
      DfDx_lo = std::max( DfDx_lo , _dfdx[ind[1]] );
      D2fDyDx_lo = std::max( D2fDyDx_lo, _d2fdxdy[ind[1]] );
    }

    DfDx_hi = oDx*(_f[ind[2]] - _f[ind[3]]);
    D2fDyDx_hi = oDx*(DfDy_hi - DfDy_lo);
    if (_f[ind[2]]==0.0) {
      DfDx_hi = std::min( DfDx_lo , _dfdx[ind[3]] );
      D2fDyDx_hi = std::min( D2fDyDx_hi, _d2fdxdy[ind[3]] );
    }
    else if (_f[ind[3]]==0.0) {
      DfDx_hi = std::max( DfDx_lo , _dfdx[ind[2]] );
      D2fDyDx_hi = std::max( D2fDyDx_hi, _d2fdxdy[ind[2]] );
    }
    
    // Use the DfDx's to get the g's
    if (_f[ind[0]]==0)
      g_lo = DfDx_lo*(dx-1.0) + _f[ind[1]];
    else
      g_lo = DfDx_lo*dx + _f[ind[0]];
    
    if (_f[ind[3]]==0)
      g_hi = DfDx_hi*(dx-1.0) + _f[ind[2]];
    else
      g_hi = DfDx_hi*dx + _f[ind[3]];

    // Use the D2fDyDx's to get DgDy
    if (_dfdx[ind[0]]==0)
      DgDy_lo = D2fDyDx_lo*(dx-1.0) + _dfdx[ind[1]];
    else
      DgDy_lo = D2fDyDx_lo*dx + _dfdx[ind[0]];

    if (_dfdx[ind[3]]==0)
      DgDy_hi = D2fDyDx_hi*(dx-1.0) + _dfdx[ind[2]];
    else
      DgDy_hi = D2fDyDx_hi*dx + _dfdx[ind[3]];
    
    DgDy = (DgDy_hi + DgDy_lo >=0 ? 1.0 : -1.0) * std::max( std::fabs(DgDy_lo), std::fabs(DgDy_hi) );
      

    // Use the g's and DgDy to get the interpolated value
    if (g_lo==0.0)
      f = DgDy*(1.0-dy) + g_hi;
    else
      f = DgDy*dy + g_lo;
  }
  else
    f = 0.0;
}

/*** Bilinear interpolation ***/
void Interpolator2D::linear(double x, double y, double& f)
{
  static int ix[4], iy[4], ind[4];

  // Get cell corners
  get_cell(x,y,ix,iy);

  // Get indicies
  for (int i=0; i<4; ++i) {
    ind[i] = index(ix[i],iy[i]);
  }

  // 1.0/step sizes
  double oDx = 1.0/(_x[ix[2]] - _x[ix[0]]); // in x direction
  double oDy = 1.0/(_y[iy[2]] - _y[iy[0]]); // in y direction

  // dx and dy from imax
  double dx, dy;
  if (ix[0]!=ix[2])
    dx = (x-_x[ix[0]])*oDx;
  else
    dx = 0.0;

  if (iy[0]!=iy[2])
    dy = (y-_y[iy[0]])*oDy;
  else
    dy = 0.0;

  f = (1.0-dx)*(1.0-dy) * _f[ind[0]]
    + dx*(1.0-dy) * _f[ind[1]]
    + dx*dy * _f[ind[2]]
    + (1.0-dx)*dy * _f[ind[3]];

}

/*** Bicubic interpolation ***/
void Interpolator2D::bicubic(double x, double y, double& f)
{
  if (!_derivatives_defined)
    fd_derivatives();

  static int ix[4], iy[4], ind[4];

  // Get cell corners
  get_cell(x,y,ix,iy);

  // Get indicies
  for (int i=0; i<4; ++i) {
    ind[i] = index(ix[i],iy[i]);
  }

  // 1.0/step sizes
  double oDx = 1.0/(_x[ix[2]] - _x[ix[0]]); // in x direction
  double oDy = 1.0/(_y[iy[2]] - _y[iy[0]]); // in y direction

  // dx and dy from imax
  double dx, dy;
  if (ix[0]!=ix[2])
    dx = (x-_x[ix[0]])*oDx;
  else
    dx = 0.0;

  if (iy[0]!=iy[2])
    dy = (y-_y[iy[0]])*oDy;
  else
    dy = 0.0;

  // If at edge, just do linear interpolation
  if (!dx || !dy) {
    f = (1.0-dx)*(1.0-dy) * _f[ind[0]]
      + dx*(1.0-dy) * _f[ind[1]]
      + dx*dy * _f[ind[2]]
      + (1.0-dx)*dy * _f[ind[3]];

    return;
  }

  // Fill bicubic arrays
  static double v[4], dvdx[4], dvdy[4], d2vdxdy[4], dfdx, dfdy;
  for (int i=0; i<4; ++i) {
    v[i] = _f[ind[i]];
    dvdx[i] = _dfdx[ind[i]];
    dvdy[i] = _dfdy[ind[i]];
    d2vdxdy[i] = _d2fdxdy[ind[i]];
  }

  //bcuint(v-1,dvdx-1,dvdy-1,d2vdxdy-1,_x[ix[0]],_x[ix[2]],_y[iy[0]],_y[iy[2]],x,y,f,dfdx,dfdy);
  bcuint(v,dvdx,dvdy,d2vdxdy,_x[ix[0]],_x[ix[2]],_y[iy[0]],_y[iy[2]],x,y,f,dfdx,dfdy);
}


/**** Bicubic spline interpolator ****/
void Interpolator2D::bicubic_spline(double x, double y, double& f)
{
  if (!_derivatives_defined)
    fd_derivatives();

  static int ix[4], iy[4], ind[4];

  // Get cell corners
  get_cell(x,y,ix,iy);

  // Get indicies
  for (int i=0; i<4; ++i)
    ind[i] = index(ix[i],iy[i]);
  

  // 1.0/step sizes
  double oDx = 1.0/(_x[ix[2]] - _x[ix[0]]); // in x direction
  double oDy = 1.0/(_y[iy[2]] - _y[iy[0]]); // in y direction

  // dx and dy from imax
  double dx, dy;
  if (ix[0]!=ix[2])
    dx = (x-_x[ix[0]])*oDx;
  else
    dx = 0.0;

  if (iy[0]!=iy[2])
    dy = (y-_y[iy[0]])*oDy;
  else
    dy = 0.0;

  // If at edge, just do linear interpolation
  if (!dx || !dy) {
    f = (1.0-dx)*(1.0-dy) * _f[ind[0]]
      + dx*(1.0-dy) * _f[ind[1]]
      + dx*dy * _f[ind[2]]
      + (1.0-dx)*dy * _f[ind[3]];

    return;
  }

  //Now do spline algorithm
  //First spline through rows to get column of values
  std::valarray<double> fx(_x.size()); 
  for ( int i = 0; i < int(_x.size()); ++i)
  {
    double a = (_y[iy[2]] - y)*oDy;
    double b = (y - _y[iy[0]])*oDy;
    fx[i] = a*_f[index(i,iy[0])] + b*_f[index(i,iy[2])] + 
            ( (a*a*a - a)*_spline_derivs[index(i,iy[0])] + 
              (b*b*b - b)*_spline_derivs[index(i,iy[2])] )/(6.0*oDy*oDy); 
  }

  //Now we need to do a column spline to find the final value
  column_spline(ix[0], ix[2], x, fx, f);

}

/* Private bicubic column spline routine */
void Interpolator2D::column_spline(int ilow, int ihi, double x, std::valarray<double> fx, double& f)
{
  //Set up 1d spline for the column we passed
  int n = _x.size();
  std::valarray<double> u(0.0,n-1), df2(0.0,n);
  for ( int i = 1; i < n-1; ++i)
  {
    double sig = (_x[i]-_x[i-1])/(_x[i+1]-_x[i-1]);
    double p = sig*fx[i-1] + 2.0;
    df2[i] = (sig-1.0)/p;
    u[i] = (fx[i+1]-fx[i])/(_x[i+1] - _x[i])
             - (fx[i] - fx[i-1])/(_x[i]-_x[i-1]);
    u[i] = ( 6.0*u[i]/(_x[i+1] - _x[i-1]) -sig*u[i-1])/p;
  }

  //Now do backsubstitution loop of tridiagonal algorithm
  for ( int i = n-2; i>=0; --i)
    df2[i] = df2[i]*df2[i+1] + u[i];
  
  //Now we have already found the correct index position so now we 
  //perform our final interpolation
  double oDx = 1.0/(_x[ihi]-_x[ilow]);
  double a = (_x[ihi]-x)*oDx;
  double b = (x-_x[ilow])*oDx;

  f = a*fx[ilow] + b*fx[ihi] + 
      ( (a*a*a - a)*df2[ilow] + 
        (b*b*b - b)*df2[ihi]   )/(6*oDx*oDx);
}


/* Private bicubic subroutines (from Numerical Recipies) */
void Interpolator2D::bcuint(double y[], double y1[], double y2[], double y12[],
			    double x1l, double x1u, double x2l, double x2u,
			    double x1, double x2, double &ansy, double &ansy1,
			    double &ansy2)
{
  int i;
  double t,u,d1,d2,c[5][5];
 
  d1=x1u-x1l;
  d2=x2u-x2l;
  bcucof(y,y1,y2,y12,d1,d2,c);
  t=(x1-x1l)/d1;
  u=(x2-x2l)/d2;
  ansy=ansy2=ansy1=0.0;
  for (i=4;i>=1;i--) {
    ansy=t*ansy + ((c[i][4]*u+c[i][3])*u+c[i][2])*u+c[i][1];
    ansy2=t*ansy2 + (3.0*c[i][4]*u+2.0*c[i][3])*u+c[i][2];
    ansy1=u*ansy1 + (3.0*c[4][i]*t+2.0*c[3][i])*t+c[2][i];
  }
  ansy1 /= d1;
  ansy2 /= d2;
}
void Interpolator2D::bcucof(double y[], double y1[], double y2[], double y12[],
			   double d1, double d2, double c[5][5])
{
  static int wt[16][16]=
  { {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
    {-3,0,0,3,0,0,0,0,-2,0,0,-1,0,0,0,0},
    {2,0,0,-2,0,0,0,0,1,0,0,1,0,0,0,0},
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,-3,0,0,3,0,0,0,0,-2,0,0,-1},
    {0,0,0,0,2,0,0,-2,0,0,0,0,1,0,0,1},
    {-3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0},
    {9,-9,9,-9,6,3,-3,-6,6,-6,-3,3,4,2,1,2},
    {-6,6,-6,6,-4,-2,2,4,-3,3,3,-3,-2,-1,-1,-2},
    {2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0},
    {-6,6,-6,6,-3,-3,3,3,-4,4,2,-2,-2,-2,-1,-1},
    {4,-4,4,-4,2,2,-2,-2,2,-2,-2,2,1,1,1,1}};
  int l,k,j,i;
  double xx,d1d2,cl[16],x[16];
 
  d1d2=d1*d2;
  for (i=1;i<=4;i++) {
    x[i-1]=y[i-1];
    x[i+3]=y1[i-1]*d1;
    x[i+7]=y2[i-1]*d2;
    x[i+11]=y12[i-1]*d1d2;
    //x[i-1]=y[i];
    //x[i+3]=y1[i]*d1;
    //x[i+7]=y2[i]*d2;
    //x[i+11]=y12[i]*d1d2;
  }
  for (i=0;i<=15;i++) {
    xx=0.0;
    for (k=0;k<=15;k++) xx += wt[i][k]*x[k];
    cl[i]=xx;
  }
  l=0;
  for (i=1;i<=4;i++)
    for (j=1;j<=4;j++) 
      c[i][j]=cl[l++];
}




};
