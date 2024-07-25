/*!
  \file interpolator1D.cpp
  \author Avery E. Broderick
  \date  April, 2017
  \brief Implements a multipurpose linear interpolation function.
  \details To be added
*/

#include "interpolator1D.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace Themis {

Interpolator1D::Interpolator1D(std::string method)
  : _x(0), _y(0), _method(method)
{
  
}
Interpolator1D::Interpolator1D(std::vector<double> x, std::vector<double> f, std::string method)
  : _x(x), _y(f), _method(method)
{
  if (_x.size() != _y.size())
  {
    std::cerr << "Interpolator1D: x and f must be same dimension!\n";
    std::exit(1);
  }
  if (method == "mcubic")
    set_mcubic();
}
Interpolator1D::Interpolator1D(std::string ifile, size_t column, size_t headers, std::string method)
  : _x(0), _y(0), _method(method)
{
  std::ifstream in(ifile.c_str());
  for (size_t i=0; i<headers; ++i)
    in.ignore(4096,'\n');

  double val;
  do {
    in >> val;
    _x.push_back(val);
    for (size_t i=1; i<column; ++i)
      in >> val;
    _y.push_back(val);
    in.ignore(4096,'\n');
  } while (!in.eof());
  _x.pop_back();
  _y.pop_back();

  
  if (_x.size() != _y.size())
  {
    std::cerr << "Interpolator1D: x and f must be same dimension!\n";
    std::exit(1);
  }
  if (_method == "mcubic")
    set_mcubic();
  /*
  // DEBUGGING
  for (size_t i=0; i<_x.size(); ++i) {
    std::cout << std::setw(15) << _x[i]
	      << std::setw(15) << _y[i]
	      << std::endl;
  }
  */
}
void Interpolator1D::set_tables(std::vector<double> x, std::vector<double> f)
{
  _x.resize(0);
  _y.resize(0);
  _x = x;
  _y = f;

  if (_x.size() != _y.size())
  {
    std::cerr << "Interpolator1D: x and f must be same dimension!\n";
    std::exit(1);
  }
  if (_method=="mcubic")
    set_mcubic();
}


double Interpolator1D::operator()(double x) const
{
  if (_method == "linear")
    return linear(x);
  else if (_method == "mcubic")
    return mcubic(x);
  else
    return 0;
}

double Interpolator1D::linear(double x) const
{
  if (x>_x[0] && x<_x[_x.size()-1]) {
    std::vector<double>::const_iterator p = std::lower_bound(_x.begin(),_x.end(),x);
    // p should now be an iterator to the first value less than x
    size_t i = p - _x.begin() - 1;
    double dx = (x-_x[i])/(_x[i+1]-_x[i]);
    return (  dx*_y[i+1] + (1.0-dx)*_y[i] );
  }
  else if (x<=_x[0])
    return _y[0];
  else
    return _y[_x.size()-1];
}


void Interpolator1D::set_mcubic()
{
  //Calculate the initial secants
  size_t n = _y.size();
  _s.resize(n-1);
  _dx.resize(n-1);
  _dy.resize(n-1);

  //Find the secant and differences for efficiency
  for ( size_t i = 0; i < n-1; ++i ){
    _dx[i] = _x[i+1]-_x[i];
    _dy[i] = _y[i+1]-_y[i];
    _s[i] = _dy[i]/_dx[i];
  }
 
  //Now find the interpolation coefficients
  _c1.resize(n,0.0);_c2.resize(n-1,0.0);_c3.resize(n-1,0.0);
  
  //First order coefficients
  _c1[0] = _s[0];
  _c1[n-1] = _s[n-2];

  for ( size_t i = 1; i < n-1; ++i )
  {
    if (_s[i]*_s[i-1] <= 0) 
      _c1[i] = 0.0;
    else
      _c1[i] = 3.0*(_dx[i]+_dx[i-1])/( (2*_dx[i]+_dx[i-1])/_s[i-1] + (_dx[i]+2.0*_dx[i-1])/_s[i]);
  
  }

  //Second and third order coefficients
  for ( size_t i = 0; i < n-1; ++i )
  {
    double dx = _dx[i];
    _c2[i] = (3*_s[i] - 2*_c1[i] - _c1[i+1])/dx;
    _c3[i] = (_c1[i] + _c1[i+1] - 2*_s[i])/(dx*dx);
  }

}

double Interpolator1D::mcubic(double x) const
{
  if (x>_x[0] && x<_x[_x.size()-1]) {
    std::vector<double>::const_iterator p = std::lower_bound(_x.begin(),_x.end(),x);
    // p should now be an iterator to the first value less than x
    size_t i = p - _x.begin() - 1;
    double dx = (x-_x[i]);
    double dx2 = dx*dx;
    //Find the cubic polynomial
    return (_y[i] + _c1[i]*dx + _c2[i]*dx2 + _c3[i]*dx2*dx);
  }
  else if (x<=_x[0])
    return _y[0]; 
  else
  {
    return _y[_x.size()-1]; 
  }

}
};
