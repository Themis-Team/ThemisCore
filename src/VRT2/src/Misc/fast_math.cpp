#include "fast_math.h"


namespace VRT2 {
  namespace FastMath {

    // exponential primatives
    //  A float expression version -- has limited domain thus better to use the (slightly) slower double precision form.
    float float_exp2(float x)
    {
      short int n = int(x);
      x -= n;
      union { float val; int32_t x; }  u;
      u.x = (float)( ((n+127)&255)<<23 );
      x *= VRT2_FAST_MATH_L2E;
      u.val *= ( (((0.176234528)*x + 0.5370958887)*x+0.997913837)*x + 0.9962189017);
      return u.val;
    }
    //  A double expression version -- beter domain (<0.4% from -300<x<300)
    double double_exp2(double x)
    {
      int n = int(x);
      x -= n;
      union { double val; int64_t x; }  u;
      u.x = (double)( ( int64_t((n+1023))&2047 )<<52 );
      x *= VRT2_FAST_MATH_L2E;
      u.val *= ( (((0.176234528)*x + 0.5370958887)*x+0.997913837)*x + 0.9962189017); 
      return u.val;
    }

    // logarithm primatives
    //  A float expression version -- has limited domain thus better to use the (slightly) slower double precision form.
    float float_log2(float val)
    {
      union { float val; int32_t x; } u = { val };
      float log_2 = (float)(((u.x >> 23) & 255) - 128);
      u.x   &= ~(255 << 23);
      u.x   += 127 << 23;
      log_2 += ((-0.34484843f) * u.val + 2.02466578f) * u.val - 0.67487759f;
      return (double(log_2));
    }
    //  A double expression version -- beter domain (<0.4% from 1e-300<x<1e300)
    double double_log2(double val)
    {
      union { double val; int64_t x; } u = { val };
      double log_2 = (double)(((u.x >> 52) & 2047) - 1024);
      u.x   &= ~(int64_t(2047) << 52);
      u.x   += int64_t(1023) << 52;
      log_2 += ((-0.34484843f) * u.val + 2.02466578f) * u.val - 0.67487759f;
      return (log_2);
    }
  };
};
  
