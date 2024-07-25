#include "sc_disk_map.h"

namespace VRT2 {
SC_DiskMap::SC_DiskMap(Metric& g, double router, double rinner)
  : StopCondition(g,router,rinner), _nswitches(-1)
{
}

/*** Condition at which propagation stops regardless of adiabaticity ***/
int SC_DiskMap::stop_condition(double y[], double dydx[])
{
  check_switch(y);

  if ( (y[1] > _rout) && (dydx[1] < 0) ) // moving into region, i.e. leaving region in - time
    return 1; // Stop
  else if (y[1] < _rin*_g.horizon())
    return 1;
  else if (_nswitches==1)
    return 1;
  else
    return 0;
}

/*** Stokes' Parameters at stopped place ***/
std::valarray<double> SC_DiskMap::IQUV(double y[])
{
  std::valarray<double> iquv(0.0,5);

  iquv[0] = _sa[0];
  iquv[1] = _sa[1];
  iquv[2] = _sb[0];
  iquv[3] = _sb[1];

  return iquv;
}

void SC_DiskMap::check_switch(double y[])
{
  // If it is a new ray, some initialization
  if (_nswitches<0) {
    _sa[0] = _sb[0] = 1.0e10;
    _sa[1] = _sb[1] = 0.0;
    _nswitches = 0;
    _xnew[0] = y[1];
    _xnew[1] = y[2];
    _xnew[2] = y[3];
    return;
  }

  // If it isn't a new ray, check switch:
  //  (a) check for sign switches of cos(theta) BEFORE setting xnew and xold!!
  double co = std::cos(_xnew[1]);
  double cn = std::cos(y[2]);
  if ( co*cn <= 0.0 ) {
    // increment the number of switches observed.
    ++_nswitches;

    // Get linear interpolation in Cartesianized coordinates
    double xo = _xnew[0]*std::sin(_xnew[1])*std::cos(_xnew[2]);
    double yo = _xnew[0]*std::sin(_xnew[1])*std::sin(_xnew[2]);
    double zo = _xnew[0]*co;

    double xn = y[1]*std::sin(y[2])*std::cos(y[3]);
    double yn = y[1]*std::sin(y[2])*std::sin(y[3]);
    double zn = y[1]*cn;

    double dz = zo/(zo-zn);

    double xz = xn*dz + xo*(1.0-dz);
    double yz = yn*dz + yo*(1.0-dz);
    double zz = zn*dz + zo*(1.0-dz);

    if (_sa[0]>1.0e3 && zo>=0.0 && zn<=0.0) { // coming from above
      _sa[0] = std::sqrt( xz*xz + yz*yz + zz*zz );
      _sa[1] = std::atan2(yz,xz);
    }
    if (_sb[0]>1.0e3 && zo<=0.0 && zn>=0.0) { // coming from below
      _sb[0] = std::sqrt( xz*xz + yz*yz + zz*zz );
      _sb[1] = std::atan2(yz,xz);
    }
  }
  else { // ONLY IF NOT STOPPING SO THAT STEPPING UP TO THE STOPPING POINT DOESN'T MAKE
         //  SPURIOUS BELOW RAYS AND SCREW UP THE STOPPING POINT!!!
    //  (A) save current _xnew into _xold and get new position
    for (size_t i=0; i<3; ++i) {
      _xold[i] = _xnew[i];
      _xnew[i] = y[i+1];
    }
  }
}
};
