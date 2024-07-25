/*************************************************/
/* Notes:
   Defines ray interface.
*/
/*************************************************/

// Only include once
#ifndef VRT2_RAY_H
#define VRT2_RAY_H

// Standard Library Header
#include <vector>
#include <valarray>
#include <string>
#include <iostream>

namespace VRT2 {
class Ray
{
 public:
  virtual ~Ray() {};

  virtual void reinitialize(FourVector<double>&,FourVector<double>&) = 0;

  // Prpagation  
  virtual std::vector<std::string> propagate(double h, std::string output="!") = 0;

  // Functions for output
  virtual std::valarray<double> IQUV() = 0;
  virtual double tau() = 0;
  virtual double D() { return 0; };
  virtual void output_ray(std::string,int) = 0;
  virtual void output_ray(std::ostream&,int) = 0;
};
};
#endif
