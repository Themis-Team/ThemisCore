/*********************************************************/
/*** Generates dirt on a disk given the output from    ***/
/*  generate_dirty_disk.cpp                              */
/*                                                       */
/* This takes a ElectronDensity object and adds the dirt */
/*                                                       */
/*********************************************************/

#ifndef VRT2_ED_DIRTY_DISK_H
#define VRT2_ED_DIRTY_DISK_H

#include "iostream"
#include "fstream"
#include "iomanip"

#include "electron_density.h"
#include <cmath>
#include <stdio.h>
using namespace std;

namespace VRT2 {
class ED_DirtyDisk : public ElectronDensity
{
 public:
  ED_DirtyDisk(ElectronDensity& ed, Metric& g, double rISCO, double amp, std::string fname, int Nlr, int Nth, int Nph);
  virtual ~ED_DirtyDisk();

  virtual double get_density(double t, double r, double theta, double phi);

 private:
  ElectronDensity& _ed_raw;
  //Metric& _g;
  double _amp;
  double _rISCO;

  double _lrmin, _lrmax;
  int _Nlr, _Nth, _Nph;
  double ***_dd;

  double _dlr, _dth, _dph;

  double _Omega;
};

inline double ED_DirtyDisk::get_density(double t,double r,double theta,double phi)
{
  double base_density = _ed_raw(t,r,theta,phi);

  if (r<_rISCO)
    return std::min(1e-10*base_density,1e-10);

  // If it shears it won't retain the structure we put on it!
  _Omega = 0.0; //1.0/( std::pow(_rISCO/_g.mass(),1.5) + _g.ang_mom() );
  phi += _Omega*t;

  phi = std::atan2(std::sin(theta)*std::cos(phi),std::sin(theta)*std::sin(phi));
  if (phi<0)
    phi += 2*M_PI;
  theta = std::acos(std::cos(theta));

  double lr = std::log(r);

  // Do interpolation
  //   1st get lower corner
  int ilr = int((lr-_lrmin)/_dlr);
  int ith = int(theta/_dth);
  int iph = int(phi/_dph);
  //   2nd check region
  if (ilr<0 || ilr>=_Nlr-1)
    return base_density;

  //   3rd do interpolation
  double dx = (lr-(ilr*_dlr+_lrmin))/_dlr;
  double dy = (theta-ith*_dth)/_dth;
  double dz = (phi-iph*_dph)/_dph;
  double mx = 1.0-dx;
  double my = 1.0-dy;
  double mz = 1.0-dz;

  int ilru = ilr+1;
  int ithu = (ith<_Nth-1 ? ith+1 : ith);
  int iphu = (iph+1)%_Nph;


  double dd = _dd[ilr  ][ith  ][iph  ] * mx*my*mz
            + _dd[ilru ][ith  ][iph  ] * dx*my*mz
            + _dd[ilr  ][ithu ][iph  ] * mx*dy*mz
            + _dd[ilr  ][ith  ][iphu ] * mx*my*dz
            + _dd[ilru ][ithu ][iph  ] * dx*dy*mz
            + _dd[ilr  ][ithu ][iphu ] * mx*dy*dz
            + _dd[ilru ][ith  ][iphu ] * dx*my*dz
            + _dd[ilru ][ithu ][iphu ] * dx*dy*dz;
  
  //double d = std::max(base_density*std::exp(_amp*dd),1e-10);
  double d = std::max(base_density*(1.0+_amp*dd),1e-10);


  return ( d );
  //return ( base_density*(1.0 + _amp*dd) );
}
};
#endif
