/////////////////
// <1% accurate reimplimentations of exp, log, and pow that are between
// 3 and 6 times faster than the standard library expressions.  This is
// obtained at the expense of some accuracy, and exception handeling.
// Therefore, these should be used only where neither are a problem.
//
// Note that these depend on the IEEE implementation of the float and
// double data types.
//
#ifndef VRT2_FAST_MATH_H
#define VRT2_FAST_MATH_H


#include <cmath>
#include <inttypes.h>

#define VRT2_FAST_MATH_L2E  0.69314718055966295651160180568695068359375
#define VRT2_FAST_MATH_OL2E 1.44269504088896340735992468100189213742664

namespace VRT2 {

  namespace FastMath {

    // exponential primatives
    float float_exp2(float x);
    double double_exp2(double x);

    // logarithm primatives
    float float_log2(float x);
    double double_log2(double x);


    // Fast exponential (<0.4% over -300 to 300)
    inline double exp(double x)
    {
#if defined(USE_FAST_MATH)
      if (x<-300)
	return 0;
      if (x>300)
	return std::exp(x);
      return (double_exp2(x*VRT2_FAST_MATH_OL2E));
#else
      return std::exp(x);
#endif
    };

    // Fast logarithm (<0.4% over 1e-300 to 1e300)
    inline double log(double x)
    {
#if defined(USE_FAST_MATH)
      return (double_log2(x)*VRT2_FAST_MATH_L2E);
#else
      return std::log(x);
#endif
    };

    // Fast pow
    inline double pow(double x, double p)
    {
#if defined(USE_FAST_MATH)
      if (x<1e-200 || x>1e200)
	return std::pow(x,p);
      else
	return (exp(p*log(x)));
#else
      return (std::pow(x,p));
#endif
    };

    // Alternative specification for integer powers to take advantage
    // of the optimized forms.
    inline double pow(double x, int p)
    {
      return (std::pow(x,p));
    };
    
  };
};

#endif
