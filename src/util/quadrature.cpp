/*!
 \file quadrature.cpp
 \author Avery Broderick
 \date April 2019
 \brief Implements a Gaussian Quadrature integration scheme with Legendre or Hermite polynomial weights
 \details See code below
*/


#include "quadrature.h"


namespace Themis{

GaussianQuadrature::GaussianQuadrature(int N)
: _N(N)
{
  // First get the weights
  _x = new double[_N];
  _v = new double[_N];
  set_weights("GaussLegendre");
}

GaussianQuadrature::~GaussianQuadrature()
{
    delete[] _x;
    delete[] _v;
}


double GaussianQuadrature::integrate(Integrand& i, double a, double b)
{
  
  double dx = (b-a);
  double x = 0;
  double sum=0.0;
  for (int j=0; j<_N; j++)
  {
    x = dx*_x[j]+a;
    sum += _v[j]*i(x);
  }
  sum *= dx;
    
  return sum;
}

void GaussianQuadrature::set_weights(std::string type)
{
  if (type=="GaussLegendre")
    gauss_legendre(0.0,1.0,_x-1,_v-1,_N);
  else if (type=="GaussHermite")
    gauss_hermite(_x-1,_v-1,_N);
  else
  {
    std::cerr << "GaussianQuadrature : Unknown classical weight type : " << type << std::endl
              << "Please select one of GaussLegendre or GaussHermite " << std::endl;
    std::exit(0);
  }
}


#define EPS 3.0e-11
void GaussianQuadrature::gauss_legendre(double x1, double x2, double x[], double w[], int n) const
{
  int m,j,i;
  double z1,z,xm,xl,pp,p3,p2,p1;
  
  m=(n+1)/2;
  xm=0.5*(x2+x1);
  xl=0.5*(x2-x1);
  for (i=1;i<=m;i++) {
    z=std::cos(M_PI*(i-0.25)/(n+0.5));
    do {
      p1=1.0;
      p2=0.0;
      for (j=1;j<=n;j++) {
	p3=p2;
	p2=p1;
	p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
      }
      pp=n*(z*p1-p2)/(z*z-1.0);
      z1=z;
      z=z1-p1/pp;
    } while (std::fabs(z-z1) > EPS);
    x[i]=xm-xl*z;
    x[n+1-i]=xm+xl*z;
    w[i]=2.0*xl/((1.0-z*z)*pp*pp);
    w[n+1-i]=w[i];
  }
}
#undef EPS
//#define EPS 3.0e-6
#define EPS 3.0e-14
#define PIM4 0.7511255444649425
#define MAXIT 100
void GaussianQuadrature::gauss_hermite(double x[], double w[], int n) const
{
  int i,its,j,m;
  double p1,p2,p3,pp,z=1,z1;

  m=(n+1)/2;
  for (i=1;i<=m;i++) {
    if (i == 1) {
      z=std::sqrt(double(2*n+1))-1.85575*std::pow(double(2*n+1),-0.16667);
    } else if (i == 2) {
      z -= 1.14*std::pow(double(n),0.426)/z;
    } else if (i == 3) {
      z=1.86*z-0.86*x[1];
    } else if (i == 4) {
      z=1.91*z-0.91*x[2];
    } else {
      z=2.0*z-x[i-2];
    }
    for (its=1;its<=MAXIT;its++) {
      p1=PIM4;
      p2=0.0;
      for (j=1;j<=n;j++) {
	p3=p2;
	p2=p1;
	p1=z*std::sqrt(2.0/j)*p2-std::sqrt(double(j-1)/j)*p3;
      }
      pp=std::sqrt(double(2*n))*p2;
      z1=z;
      z=z1-p1/pp;
      if (std::fabs(z-z1) <= EPS) break;
    }
    if (its > MAXIT)
    {
      std::cerr << "GaussianQuadrature:: gauss_hermite : Too many iterations in guass_hermite\n";
      std::cerr << "std::fabs(z-z1) = " << std::fabs(z-z1) << '\n';
      std::exit(0);
    }
    x[i]=z;
    x[n+1-i] = -z;
    w[i]=2.0/(pp*pp);
    w[n+1-i]=w[i];
  }


  // Renormalize so we have v's not w's
  for (i=1;i<=n;i++)
    w[i] /= std::exp(-x[i]*x[i]);
}
#undef EPS
#undef PIM4
#undef MAXIT

};

