// Various random number generators

#ifndef VRT2_RANDOM_NUMBER_GENERATOR_H
#define VRT2_RANDOM_NUMBER_GENERATOR_H

#include <cmath>
#include <cstdlib>

// Interface
namespace VRT2 {
class RandomNumberGenerator
{
 public:
  virtual ~RandomNumberGenerator() {};
  virtual double rand()=0;
};
// System supplied generator
class SystemSuppliedRNG : public RandomNumberGenerator
{
 public:
  SystemSuppliedRNG(int seed);
  virtual ~SystemSuppliedRNG() {};
  virtual double rand();
};
// Park and Miller "Minimal Standard" (a la NR)
class MinimalStandardRNG : public RandomNumberGenerator
{
 public:
  MinimalStandardRNG(int seed);
  virtual ~MinimalStandardRNG() {};
  virtual double rand();
 private:
  int _seed;
  int _IA, _IM, _IQ, _IR, _MASK;
  double _AM;
};
// L'Ecuyer with Bays-Durham shuffle (a la NR)
class Ran2RNG : public RandomNumberGenerator
{
 public:
  Ran2RNG(int seed);
  virtual ~Ran2RNG();
  virtual double rand();
 private:
  int _idum, _idum2;
  int _IM1, _IM2, _IMM1, _IA1, _IA2, _IQ1, _IQ2;
  int _IR1, _IR2, _NTAB, _NDIV;
  double _AM, _EPS, _RNMX;
  int _iy;
  int *_iv;
};
// Gaussian random variable with zero mean and unit variance
template<class T>
class GaussianRandomNumberGenerator : public RandomNumberGenerator
{
 public:
  GaussianRandomNumberGenerator(int seed);
  virtual ~GaussianRandomNumberGenerator() {};
  virtual double rand();

 private:
  bool _compute_new;
  double _r1, _r2;
  T _rng;
};
// Poisson random variable generator
template<class T>
class PoissonRandomNumberGenerator : public RandomNumberGenerator
{
 public:
  PoissonRandomNumberGenerator(int seed, double mean);
  virtual ~PoissonRandomNumberGenerator() {};
  virtual double rand();

  double prob(int n); // Probability (just for check)

 private:
  T _rng;
  double _mean;
  double _sq, _alxm, _g;
  double gammln(double xx);
  double factrl(int n);
};

// Box-Muller Gaussian Random Variables
template<class T>
GaussianRandomNumberGenerator<T>::GaussianRandomNumberGenerator(int seed)
: _compute_new(true), _rng(seed)
{
}

template<class T>
double GaussianRandomNumberGenerator<T>::rand()
{
  if (_compute_new)
  {
    _compute_new = false;
    double x1, x2;
    x1 = std::sqrt( -2.0*std::log(_rng.rand()) );
    x2 = 2.0*M_PI*_rng.rand();

    _r1 = x1*cos(x2);
    _r2 = x1*sin(x2);

    return _r1;
  }
  else
  {
    _compute_new = true;

    return _r2;
  }
}

// NR Rejection for generating Poisson Random Variables
template<class T>
PoissonRandomNumberGenerator<T>::PoissonRandomNumberGenerator(int seed, double mean)
  : _rng(seed), _mean(mean)
{
  if (_mean<12.0)
  {
    _g = std::exp(-_mean);
  }
  else
  {
    _sq = std::sqrt(2.0*_mean);
    _alxm = std::log(_mean);
    _g = _mean*_alxm-gammln(_mean+1.0);
  }
}
template<class T>
double PoissonRandomNumberGenerator<T>::rand()
{
  //static double sq,alxm;
  double em,t,y;

  if (_mean < 12.0)
  {
    em = -1;
    t=1.0;
    do {
      ++em;
      t *= _rng.rand();
    } while (t > _g);
  }
  else
  {
    do {
      do {
	y=std::tan(M_PI*_rng.rand());
	em=_sq*y+_mean;
      } while (em < 0.0);
      em=std::floor(em);
      t=0.9*(1.0+y*y)*std::exp(em*_alxm-gammln(em+1.0)-_g);
    } while (_rng.rand() > t);
  }

  return em;
}
template<class T>
double PoissonRandomNumberGenerator<T>::gammln(double xx)
{
  double x,y,tmp,ser;
  static double cof[6]={76.18009172947146,-86.50532032941677,
			24.01409824083091,-1.231739572450155,
			0.1208650973866179e-2,-0.5395239384953e-5};
  int j;

  y=x=xx;
  tmp=x+5.5;
  tmp -= (x+0.5)*std::log(tmp);
  ser=1.000000000190015;

  for (j=0;j<=5;j++)
    ser += cof[j]/++y;

  return -tmp+log(2.5066282746310005*ser/x);
}
template<class T>
double PoissonRandomNumberGenerator<T>::factrl(int n)
{
  static int ntop=4;
  static double a[33]={1.0,1.0,2.0,6.0,24.0};
  int j;

  if (n > 32)
    return std::exp(gammln(n+1.0));

  while (ntop<n) {
    j=ntop++;
    a[ntop]=a[j]*ntop;
  }
  return a[n];
}
template<class T>
double PoissonRandomNumberGenerator<T>::prob(int n)
{
  return std::pow(_mean,n)*std::exp(-_mean)/factrl(n);
}


};
#endif
