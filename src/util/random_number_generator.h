/*!
  \file random_number_generator.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for various random number generators using the algorithms
   in "Numerical Recipes in C" by Flannery, Teukolsky, Press, and Vettering.
  \details To be added
*/

#ifndef Themis_RANDOM_NUMBER_GENERATOR_H
#define Themis_RANDOM_NUMBER_GENERATOR_H

#include <cmath>
#include <cstdlib>

namespace Themis {

/*! 
  \brief Defines an interface for random number generators.
  \details 
  \warning Contains purely virtual functions and thus only
  child classes can be instantiated explicitly.
*/
class RandomNumberGenerator
{
 public:
  virtual ~RandomNumberGenerator() {};
  //! Returns a random number
  virtual double rand()=0;
};


/*! 
  \brief A wrapper for the system supplied generator.
  \details 
*/
class SystemSuppliedRNG : public RandomNumberGenerator
{
 public:
  //! Constructor for the wrapped the system supplied generator,
  //! takes an integer seed that specifies the full random sequence.
  SystemSuppliedRNG(int seed);
  virtual ~SystemSuppliedRNG() {};
  virtual double rand();
};

/*! 
  \brief The Park and Miller "Minimal Standard" (a la NR).
  \details 
*/
class MinimalStandardRNG : public RandomNumberGenerator
{
 public:
  //! Constructor for the minimal standard generator, takes an integer seed
  //! that specifies the full random sequence.
  MinimalStandardRNG(int seed);
  virtual ~MinimalStandardRNG() {};
  virtual double rand();
 private:
  int _seed;
  int _IA, _IM, _IQ, _IR, _MASK;
  double _AM;
};

/*! 
  \brief The L'Ecuyer with Bays-Durham shuffle (a la NR)
  \details 
*/
class Ran2RNG : public RandomNumberGenerator
{
 public:
  //! Constructor for L'Ecuyer with Bays-Durham shuffle, takes an integer seed
  //! that specifies the full random sequence.
  Ran2RNG(int seed);
  virtual ~Ran2RNG();
  virtual void reset_seed(int seed);
  virtual double rand();
 private:
  int _idum, _idum2;
  int _IM1, _IM2, _IMM1, _IA1, _IA2, _IQ1, _IQ2;
  int _IR1, _IR2, _NTAB, _NDIV;
  double _AM, _EPS, _RNMX;
  int _iy;
  int *_iv;
};
 
/*! 
  \brief Defines unit variance, zero mean Gaussian random variable template.
  \details Takes a unit variat random number generator class as a template argument.
  Uses the Box-Muller transformation.
*/
template<class T>
class GaussianRandomNumberGenerator : public RandomNumberGenerator
{
 public:
  //! Constructor for Gassian random number generator, takes an integer seed
  //! that specifies the full random sequence.
  GaussianRandomNumberGenerator(int seed);
  virtual ~GaussianRandomNumberGenerator() {};
  virtual double rand();

 private:
  bool _compute_new;
  double _r1, _r2;
  T _rng;
};


/*! 
  \brief Defines a random number generator with a Poisson distribution.
  \details Takes a unit variat random number generator class as a template
   argument. Uses the rejection method (a la NR).
*/
template<class T>
class PoissonRandomNumberGenerator : public RandomNumberGenerator
{
 public:
  //! Constructor for Gassian random number generator, takes an integer seed
  //! that specifies the full random sequence and a mean for the Poisson distribution.
  PoissonRandomNumberGenerator(int seed, double mean);
  virtual ~PoissonRandomNumberGenerator() {};
  virtual double rand();

  double prob(int n); //!< Returns probability (just for check)

 private:
  T _rng;
  double _mean;
  double _sq, _alxm, _g;
  double gammln(double xx);
  double factrl(int n); 
};

//----------------------------- Implemetations of template cases -------------

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
