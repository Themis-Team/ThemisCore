/********************************************/
/****** Header file for fourvector.cpp ******/
/****** Contains Four Vector Base Class *****/
/* NOTES:
   Needs to be initialized by k.cov or k.con.
   Construction requires a metric.
   Functions require that the metric has been initialized.
*/
/*******************************************/

// Some optimization options that must be set at library compile time
// These sepecify the structure of the metric and can result in 
// substantial time savings.
//#define DIAGONAL_METRIC 
//#define BOYER_LINDQUIST_LIKE_METRIC  // Assumes only off-diagonal element is g_03

//#define OPENMP_PRAGMA_VECTORIZE




// Only include once
#ifndef VRT2_FOURVECTOR_H
#define VRT2_FOURVECTOR_H

// Standard Library Headers
#include <iostream>
#include <iomanip>
#include <valarray>

// Local Headers
#include "metric.h"

namespace VRT2 {

template<class T>
class FourVector{

 public:
  // Constructor and Destructor
  FourVector();
  FourVector(Metric& metric);
  FourVector(Metric& metric, double x);
  FourVector(const FourVector& w);

  // Set gptr manually
  void set_gptr(Metric* metric) { _gptr = metric; };
  void set_gptr(Metric& metric) { _gptr = &metric; };
  Metric* gptr() const { return _gptr; };

  // Initialization functions
  void mkcov(const std::valarray<T>&);
  void mkcon(const std::valarray<T>&);
  // Initializes by the covariant via array
  void mkcov(const double[]);
  // Initializes by the contravariant via array
  void mkcon(const double[]);
  void mkcov(T, T, T, T); // via elements
  void mkcon(T, T, T, T); // via elements
  void mkcov(T); // Initialize all covariant elements to T
  void mkcon(T); // Initialize all contravariant elements to T
  // Return values
  T cov(int); // Returns int^th covariant element
  T con(int); // Returns int^th contravariant element
  //std::valarray<T>& cov(); // Returns vector of cov components
  //std::valarray<T>& con(); // Returns vector of con components
  std::valarray<T> cov(); // Returns vector of cov components
  std::valarray<T> con(); // Returns vector of con components


  // Index manipulations
  void raise();
  void lower();

  // Check for defined quantities (|2=cov,|3=con)
  //unsigned int defined;
  std::bitset<2> defined_list;
  //bool defined_cov;
  //bool defined_con;


  // Assigment Operator
  FourVector<T>& operator=(FourVector<T>&);

  // Unary Operators
  FourVector<T>& operator+=(FourVector<T>&);
  FourVector<T>& operator-=(FourVector<T>&);
  FourVector<T>& operator*=(double);
  FourVector<T>& operator/=(double);

  // Binary Operators
  // Addition
  template<class X>
    friend FourVector<X>& operator+(FourVector<X>&, FourVector<X>&);
  // Subtraction
  template<class X>
    friend FourVector<X>& operator-(FourVector<X>&, FourVector<X>&);
  // Scalar Multiplication
  template<class X>
    friend FourVector<X>& operator*(X, FourVector<X>&);

  /* NOTE: NEVER DIVIDE! (1/scalar)*FourVector -> 1 division 
     and 4 multiplications; MUCH faster than FourVector/scalar -> 4 
     divisions! */

  // Dot Product
  template<class X>
    friend X operator*(FourVector<X>&, FourVector<X>&);

  // epsilon*a,b,c
  template<class X>
    friend FourVector<X>& cross_product(FourVector<X>&, FourVector<X>&, FourVector<X>&);

  // Output (for debugging)
  template<class X>
    friend std::ostream& operator<<(std::ostream&, FourVector<X>&);

  //private:
  // Metric pointer
  Metric* _gptr;

  // Local vector components
  //std::valarray<T> _cov; // inside covariant piece
  //std::valarray<T> _con; // inside contravariant piece
  T _cov[4];
  T _con[4];

};

template<class T>
FourVector<T>::FourVector() 
  : _gptr(0) //, _cov(0.0,4), _con(0.0,4) 
{ 
  defined_list.reset();
}

template<class T>
FourVector<T>::FourVector(Metric& metric) 
  : _gptr(&metric) //, _cov(0.0,4), _con(0.0,4)
{
  defined_list.reset();
}

template<class T>
FourVector<T>::FourVector(Metric& metric, double x) 
  : _gptr(&metric) //, _cov(0.0,4), _con(0.0,4)
{
  defined_list.reset();
}

template<class T>
FourVector<T>::FourVector(const FourVector& w) 
  : _gptr(w.gptr())
{
  defined_list = w.defined_list;
  //_cov = w._cov;
  //_con = w._con;

#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
  {
    _cov[i] = w._cov[i];
    _con[i] = w._con[i];
  }

}

// Initialize by covariant components via std::valarray
template<class T>
inline void FourVector<T>::mkcov(const std::valarray<T>& v)
{
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _cov[i] = v[i];

  defined_list = std::bitset<2>("01");
}

// Initialize by contravariant components via std::valarray
template<class T>
inline void FourVector<T>::mkcon(const std::valarray<T>& v)
{
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _con[i] = v[i];

  defined_list = std::bitset<2>("10");
}

// Initialize by covariant components via array
template<class T>
inline void FourVector<T>::mkcov(const double v[])
{
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i1=0;i1<4;++i1)
    _cov[i1] = v[i1];

  defined_list = std::bitset<2>("01");
}

// Initialize by contravariant components via array
template<class T>
inline void FourVector<T>::mkcon(const double v[])
{
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i1=0;i1<4;++i1)
    _con[i1] = v[i1];

  defined_list = std::bitset<2>("10");
}


// Initialize by covariant components via elements
template<class T>
inline void FourVector<T>::mkcov(T v0, T v1, T v2, T v3)
{
  _cov[0] = v0;
  _cov[1] = v1;
  _cov[2] = v2;
  _cov[3] = v3;

  defined_list = std::bitset<2>("01");
}

// Initialize by contravariant components via elements
template<class T>
inline void FourVector<T>::mkcon(T v0, T v1, T v2, T v3)
{
  _con[0] = v0;
  _con[1] = v1;
  _con[2] = v2;
  _con[3] = v3;

  defined_list = std::bitset<2>("10");
}

// Initialize by covariant components via single element
template<class T>
inline void FourVector<T>::mkcov(T v)
{
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _cov[i] = v;

  defined_list = std::bitset<2>("01");
}

// Initialize by contravariant components via single element
template<class T>
inline void FourVector<T>::mkcon(T v)
{
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _con[i] = v;

  defined_list = std::bitset<2>("10");
}

// Return covariant components
template<class T>
inline T FourVector<T>::cov(int i)
{
  if(!defined_list[0]){
    lower();
    defined_list.set(0);
  }
  return (_cov[i]);
}

// Return contravariant components
template<class T>
inline T FourVector<T>::con(int i)
{
  if(!defined_list[1]){
    raise();
    defined_list.set(1);
  }
  return (_con[i]);
}

// Return covariant vector
template<class T>
  //inline std::valarray<T>& FourVector<T>::cov()
inline std::valarray<T> FourVector<T>::cov()
{
  if(!defined_list[0]){
    lower();
    defined_list.set(0);
  }
  //return (_cov);
  //std::valarray<T> c(_cov,4);
  //return c;
  return (std::valarray<T>(_cov,4));
}

// Return contravariant vector
template<class T>
  //inline std::valarray<T>& FourVector<T>::con()
inline std::valarray<T> FourVector<T>::con()
{
  if(!defined_list[1]){
    raise();
    defined_list.set(1);
  }
  //return (_con);
  //std::valarray<T> c(_con,4);
  //return c;
  return (std::valarray<T>(_con,4));
}

/*** Useful Functions ***/
// Makes covariant vector from a contravariant vector
template<class T>
inline void FourVector<T>::lower()
{
  _gptr->g(0);
#if defined(BOYER_LINDQUIST_LIKE_METRIC)
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _cov[i] = _gptr->_g[i] * _con[i];
  _cov[0] += _gptr->_g[4]*_con[3];
  _cov[3] += _gptr->_g[4]*_con[0];
#elif defined(DIAGONAL_METRIC)
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _cov[i] = _gptr->_g[i] * _con[i];
#else
  //_cov = 0.0;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _cov[i] = 0.0;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i1=0;i1<_gptr->Ng;++i1){
    _cov[_gptr->gi[i1]] += _gptr->_g[i1]*_con[_gptr->gj[i1]];
  }
#endif
}

// Makes contravariant vector from a covariant vector
template<class T>
inline void FourVector<T>::raise()
{
  _gptr->ginv(0);
#if defined(BOYER_LINDQUIST_LIKE_METRIC)
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _con[i] = _gptr->_ginv[i] * _cov[i];
  _con[0] += _gptr->_ginv[4]*_cov[3];
  _con[3] += _gptr->_ginv[4]*_cov[0];
#elif defined(DIAGONAL_METRIC)
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _con[i] = _gptr->_ginv[i] * _cov[i];
#else
  //_con = 0.0;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
    _con[i] = 0.0;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i1=0;i1<_gptr->Ng;++i1){
    _con[_gptr->gi[i1]] += _gptr->_ginv[i1]*_cov[_gptr->gj[i1]];
  }
#endif
}

/*** ASSIGNMENT OPERATOR ***/
template<class T>
inline FourVector<T>& FourVector<T>::operator=(FourVector<T>& w)
{
  _gptr = w.gptr();

  defined_list = w.defined_list;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<4; ++i)
  {
    _cov[i] = w._cov[i];
    _con[i] = w._con[i];
  }
  
  return (*this);
}



/*** UNARY OPERATIONS ***/
template<class T>
inline FourVector<T>& FourVector<T>::operator+=(FourVector<T>& w)
{
  if (defined_list[0]){
    //_cov += w.cov();
    w.cov(0);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      _cov[i] += w._cov[i];
    defined_list = std::bitset<2>("01");
  }
  else{
    //_con += w.con();
    w.con(0);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      _con[i] += w._con[i];
  }

  return (*this);
}

template<class T>
inline FourVector<T>& FourVector<T>::operator-=(FourVector<T>& w)
{
  if (defined_list[0]){
    //_cov -= w.cov();
    w.cov(0);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      _cov[i] -= w._cov[i];
    defined_list = std::bitset<2>("01");
  }
  else{
    //_con -= w.con();
    w.con(0);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      _con[i] -= w._con[i];
  }

  return (*this);
}

template<class T>
inline FourVector<T>& FourVector<T>::operator*=(double a)
{
  if (defined_list[0]){
    //_cov *= a;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      _cov[i] *= a;
    defined_list = std::bitset<2>("01");
  }
  else{
    //_con *= a;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      _con[i] *= a;
  }

  return (*this);
}

template<class T>
inline FourVector<T>& FourVector<T>::operator/=(double a)
{
  a = 1.0/a;

  if (defined_list[0]){
    //_cov *= a;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      _cov[i] *= a;
    defined_list = std::bitset<2>("01");
  }
  else{
    //_con *= a;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      _con[i] *= a;
  }

  return (*this);
}

/*** MEMORY MANAGEMENT ***/
template<class T>
inline FourVector<T>& next_fv_temp(Metric& g)
{
  static FourVector<T> temps[32];
  static unsigned int n_temp=0;
  
  n_temp = (n_temp+1)%32;
  temps[n_temp].set_gptr(g);
  return ( temps[n_temp] );
}

/*** BINARY OPERATIONS ***/
// Addition (covariant by default unless v is only contravariant)
template<class T>
inline FourVector<T>& operator+(FourVector<T>& v, FourVector<T>& w)
{
  FourVector<T> &x = next_fv_temp<T>(*(v.gptr()));
 
  if (v.defined_list[0]){
    //x._cov = v._cov + w.cov();
    w.cov(0);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      x._cov[i] = v._cov[i] + w._cov[i];

    x.defined_list = std::bitset<2>("01");
  }
  else{
    //x._con = v._con + w.con();
    w.con(0);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      x._con[i] = v._con[i] + w._con[i];

    x.defined_list = std::bitset<2>("10");
  }

  return (x);
}

// Subtraction (covariant by default unless v is only contravariant)
template<class T>
inline FourVector<T>& operator-(FourVector<T>& v, FourVector<T>& w)
{
  FourVector<T> &x = next_fv_temp<T>(*(v.gptr()));

  if (v.defined_list[0]){
    //x._cov = v._cov - w.cov();
    w.cov(0);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      x._cov[i] = v._cov[i] - w._cov[i];

    x.defined_list = std::bitset<2>("01");
  }
  else{
    //x._con = v._con - w.con();
    w.con(0);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      x._con[i] = v._con[i] - w._con[i];

    x.defined_list = std::bitset<2>("10");
  }

  return (x);
}

// Scalar Multiplication (covariant by default unless v is only contravariant)
template<class T>
inline FourVector<T>& operator*(T s, FourVector<T>& v)
{
  FourVector<T> &x = next_fv_temp<T>(*(v.gptr()));

  if (v.defined_list[0]){
    //x._cov = s * v._cov;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      x._cov[i] = s*v._cov[i];
    x.defined_list = std::bitset<2>("01");
  }
  else{
    //x._con = s * v._con;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<4; ++i)
      x._con[i] = s*v._con[i];
    x.defined_list = std::bitset<2>("10");
  }

  return (x);
}

// Dot Product
// for references
template<class T>
T operator*(FourVector<T>& v, FourVector<T>& w)
{  
  std::bitset<2> vwor = v.defined_list | w.defined_list;
  T dot_prod=0;

  // Both defined as covariant
  if (vwor==std::bitset<2>("01"))
  {
    v.gptr()->ginv(0);
#if defined(BOYER_LINDQUIST_LIKE_METRIC)
  dot_prod = 
      v.gptr()->_ginv[0]*v._cov[0]*w._cov[0]
    + v.gptr()->_ginv[1]*v._cov[1]*w._cov[1]
    + v.gptr()->_ginv[2]*v._cov[2]*w._cov[2]
    + v.gptr()->_ginv[3]*v._cov[3]*w._cov[3]
    + v.gptr()->_ginv[4]*(v._cov[3]*w._cov[0]+v._cov[0]*w._cov[3]);
#elif defined(DIAGONAL_METRIC)
  dot_prod = 
      v.gptr()->_ginv[0]*v._cov[0]*w._cov[0]
    + v.gptr()->_ginv[1]*v._cov[1]*w._cov[1]
    + v.gptr()->_ginv[2]*v._cov[2]*w._cov[2]
    + v.gptr()->_ginv[3]*v._cov[3]*w._cov[3];
#else
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i1=0;i1<v.gptr()->Ng;++i1)
    {
      dot_prod += v.gptr()->_ginv[i1]*v._cov[v.gptr()->gi[i1]]*w._cov[v.gptr()->gj[i1]];
    }
#endif
  }
  // Both defined as contravariant
  else if (vwor==std::bitset<2>("10"))
  {
    v.gptr()->g(0);
#if defined(BOYER_LINDQUIST_LIKE_METRIC)
  dot_prod = 
      v.gptr()->_g[0]*v._con[0]*w._con[0]
    + v.gptr()->_g[1]*v._con[1]*w._con[1]
    + v.gptr()->_g[2]*v._con[2]*w._con[2]
    + v.gptr()->_g[3]*v._con[3]*w._con[3]
    + v.gptr()->_g[4]*(v._con[3]*w._con[0]+v._con[0]*w._con[3]);
#elif defined(DIAGONAL_METRIC)
  dot_prod = 
      v.gptr()->_g[0]*v._con[0]*w._con[0]
    + v.gptr()->_g[1]*v._con[1]*w._con[1]
    + v.gptr()->_g[2]*v._con[2]*w._con[2]
    + v.gptr()->_g[3]*v._con[3]*w._con[3];
#else
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd reduction(+:dot_prod)
#endif
    for (int i1=0;i1<v.gptr()->Ng;++i1)
    {
      dot_prod += v.gptr()->_g[i1]*v._con[v.gptr()->gi[i1]]*w._con[v.gptr()->gj[i1]];
    }
#endif
  }
  // v covariant and w contravariant
  else if (v.defined_list[0] && w.defined_list[1])
  {
    //dot_prod = ( v._cov * w._con ).sum();
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd reduction(+:dot_prod)
#endif
    for (int i=0; i<4; ++i)
      dot_prod += v._cov[i] * w._con[i];
  }
  // w covariant and v contravariant (all others)
  else
  {
    //dot_prod = ( v._con * w._cov ).sum();
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd reduction(+:dot_prod)
#endif
    for (int i=0; i<4; ++i)
      dot_prod += v._con[i] * w._cov[i];
  }

  return (dot_prod);
}

// Cross Product
template<class T>
FourVector<T>& cross_product(FourVector<T>& a, FourVector<T>& b, 
			     FourVector<T>& c)
{
  FourVector<T> &w = next_fv_temp<T>(*(a.gptr())); // Return value

  // Make sure that covariant things are defined and then access elements
  if (!a.defined_list[0])
    a.lower();
  if (!b.defined_list[0])
    b.lower();
  if (!c.defined_list[0])
    c.lower();
    
  w.mkcon( // t
	   a._cov[1]*(b._cov[2]*c._cov[3] - b._cov[3]*c._cov[2])
	   +
	   a._cov[2]*(b._cov[3]*c._cov[1] - b._cov[1]*c._cov[3])
	   +
	   a._cov[3]*(b._cov[1]*c._cov[2] - b._cov[2]*c._cov[1]),
	   // r
	   a._cov[3]*(b._cov[2]*c._cov[0] - b._cov[0]*c._cov[2])
	   +
	   a._cov[2]*(b._cov[0]*c._cov[3] - b._cov[3]*c._cov[0])
	   +
	   a._cov[0]*(b._cov[3]*c._cov[2] - b._cov[2]*c._cov[3]),
	   // theta
	   a._cov[0]*(b._cov[1]*c._cov[3] - b._cov[3]*c._cov[1])
	   +
	   a._cov[1]*(b._cov[3]*c._cov[0] - b._cov[0]*c._cov[3])
	   +
	   a._cov[3]*(b._cov[0]*c._cov[1] - b._cov[1]*c._cov[0]),
	   // phi
	   a._cov[2]*(b._cov[1]*c._cov[0] - b._cov[0]*c._cov[1])
	   +
	   a._cov[1]*(b._cov[0]*c._cov[2] - b._cov[2]*c._cov[0])
	   +
	   a._cov[0]*(b._cov[2]*c._cov[1] - b._cov[1]*c._cov[2])
	   );



  /*
  if (a.defined%2)
    a.lower();
  if (b.defined%2)
    b.lower();
  if (c.defined%2)
    c.lower();
  */

  /*
  w.mkcon( a.cov(1)*b.cov(2)*c.cov(3) 
	   + a.cov(2)*b.cov(3)*c.cov(1)
	   + a.cov(3)*b.cov(1)*c.cov(2)
	   - a.cov(3)*b.cov(2)*c.cov(1)
	   - a.cov(2)*b.cov(1)*c.cov(3)
	   - a.cov(1)*b.cov(3)*c.cov(2),
	   
	   a.cov(3)*b.cov(2)*c.cov(0) 
	   + a.cov(2)*b.cov(0)*c.cov(3)
	   + a.cov(0)*b.cov(3)*c.cov(2)
	   - a.cov(0)*b.cov(2)*c.cov(3)
	   - a.cov(2)*b.cov(3)*c.cov(0)
	   - a.cov(3)*b.cov(0)*c.cov(2),

	   a.cov(0)*b.cov(1)*c.cov(3) 
	   + a.cov(1)*b.cov(3)*c.cov(0)
	   + a.cov(3)*b.cov(0)*c.cov(1)
	   - a.cov(3)*b.cov(1)*c.cov(0)
	   - a.cov(1)*b.cov(0)*c.cov(3)
	   - a.cov(0)*b.cov(3)*c.cov(1),

	   a.cov(2)*b.cov(1)*c.cov(0) 
	   + a.cov(1)*b.cov(0)*c.cov(2)
	   + a.cov(0)*b.cov(2)*c.cov(1)
	   - a.cov(0)*b.cov(1)*c.cov(2)
	   - a.cov(1)*b.cov(2)*c.cov(0)
	   - a.cov(2)*b.cov(0)*c.cov(1) );
  */

  w *= (1.0/a.gptr()->detg());

  return ( w );
}

template<class T>
std::ostream& operator<<(std::ostream& os, FourVector<T>& v)
{
  os.setf(std::ios::scientific);
  os << std::setprecision(3);

  // Output contravariant vector
  if (v.defined_list[1])
  {
    for (int i1=0;i1<4;++i1){
      if (i1)
	os << "  "; // Spacer
      os << "v^" << i1 << " = " 
	 << std::setw(10) << v._con[i1];
      //<< std::setw(10) << v.con(i1);
    }
    os << std::endl;
  }

  // Output covariant vector
  if (v.defined_list[0])
  {
    for (int i1=0;i1<4;++i1){
      if (i1)
	os << "  "; // Spacer
      os << "v_" << i1 << " = " 
	 << std::setw(10) << v._cov[i1];
      //<< std::setw(10) << v.cov(i1);
    }
    os << std::endl;
  }

  return os;
}

};
#endif




