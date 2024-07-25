/***************************************************/
/*** PROVIDES AN INTERPOLATION FUNCTION          ***/
/***************************************************/

#ifndef VRT2_INTERPOLATOR1D_H
#define VRT2_INTERPOLATOR1D_H

#include <string>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <math.h>
using namespace std;
#include <algorithm>

namespace VRT2 {
class Interpolator1D
{
 public:
  Interpolator1D();
  Interpolator1D(std::vector<double> x, std::vector<double> f);
  Interpolator1D(std::string ifile, size_t column, size_t headers=0);

  void set_tables(std::vector<double> x, std::vector<double> f);

  double operator()(double x) const;

 private:
  std::vector<double> _x, _y;
};
};
#endif
