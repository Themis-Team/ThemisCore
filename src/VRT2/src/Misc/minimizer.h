///////////////////////
//
// Minimizes a multi-dimensional function
//
#ifndef VRT2_MINIMIZER_H
#define VRT2_MINIMIZER_H

#include <cstdlib>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace VRT2 {
class Function
{
 public:
  virtual ~Function() {};
  virtual double operator()(std::vector<double>)=0;
  virtual size_t size() const=0; // Dimension of arguments taken by function
};

// First by doing a refining grid search
class GridMinimizer
{
 public:
  GridMinimizer() {};
  double Minimize(Function& F, std::vector<double>& xmin, std::vector<double>& xmax, std::vector<size_t> Nx, std::vector<double>& xm);
  
 private:
  std::vector<size_t> ND_from_1D(std::vector<size_t> Nx, size_t i1D) const;
};


// Powell's Method (NR)
class PowellMinimizer
{
 public:
  PowellMinimizer() {};
  double Minimize(Function& F, std::vector<double>& xm, double tol, std::vector<double>& stps);

 private:
  Function* _Fptr;
  double *_pcom, *_xicom;
  double _tol;

  void nrerror(std::string s) { std::cerr << s; std::exit(1); };

  double f1dim(double x);
  double linmin(double p[], double xi[]);
  void mnbrak(double& ax, double& bx, double& cx, double& fa, double& fb, double& fc);
  double brent(double ax, double bx, double cx, double tol, double& xmin);
  double powell(double p[], double **xi, double ftol, int& iter);
};
};
#endif

