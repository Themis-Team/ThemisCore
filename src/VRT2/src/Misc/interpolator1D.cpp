#include "interpolator1D.h"

namespace VRT2 {
Interpolator1D::Interpolator1D()
  : _x(0), _y(0)
{
}
Interpolator1D::Interpolator1D(std::vector<double> x, std::vector<double> f)
  : _x(x), _y(f)
{
}
Interpolator1D::Interpolator1D(std::string ifile, size_t column, size_t headers)
  : _x(0), _y(0)
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
}
double Interpolator1D::operator()(double x) const
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
};
