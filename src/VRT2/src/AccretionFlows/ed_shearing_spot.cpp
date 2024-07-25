#include "ed_shearing_spot.h"

namespace VRT2 {

ED_shearing_spot::ED_shearing_spot( Metric& g, double density_scale, double rSpot, double rISCO, AFV_Keplerian& afv, double t0, double r0, double theta0, double phi0)
  : _g(g), _density_scale(density_scale), _rspot(rSpot), _rISCO(rISCO), _afv(afv), _xspot_center0(0.0,4)
{

  //Set spot center
  _xspot_center0[0] = t0;
  _xspot_center0[1] = r0;
  _xspot_center0[2] = theta0;
  _xspot_center0[3] = phi0;

  //Set spot center velocity
  _g.reset(_xspot_center0);
  _uCenter0 = _afv(t0,r0,theta0,phi0);
}


double ED_shearing_spot::get_density(const double t, const double r, const double theta, const double phi)
{
  //Find initial spot location
  double xspot[4];
  xspot[0] = t;
  xspot[1] = r;
  xspot[2] = theta;
  xspot[3] = phi;

 //Find the 4-velocity for the observer
  _g.reset(xspot);
  FourVector<double> u(_g);
  u = _afv(xspot);
 
  //Now find the initial position of the spot
  double xspot0[4];

  xspot0[0] = t - u.con(0)/_uCenter0.con(0)*(t - _xspot_center0[0]);
  xspot0[1] = r;
  xspot0[2] = theta;
  xspot0[3] = phi - u.con(3)/_uCenter0.con(0)*(t - _xspot_center0[0]);

 //In general will need the determinant
  _g.reset(xspot0);
  //Get the density
  double dr = delta_r(xspot0[0], xspot0[1], xspot0[2], xspot0[3]);
  double rho = _density_scale*FastMath::exp(-dr*dr/(2.*_rspot*_rspot));
  
  //Reset ray so that it at r
  _g.reset(t,r,theta,phi);
  //Check if time is before the spot has been initialized in the observers time
  if ( t < xspot0[0])
  {
    //std::cerr << "spot off\n";
    return 1e-50;
  }
  //Don't trust stuff inside isco so damp it
  else if ( r <=  _rISCO )
    return rho*std::exp(-(r-_rISCO)*(r-_rISCO)/(0.1*_rspot*_rspot))+ 1e-50; 
  //Spot is on! Return "Gaussian spot"
  else
  {
     return rho + 1e-50; //Return proper density
  }
}



double ED_shearing_spot::delta_r(const double t, const double r, const double theta, const double phi)
{
  //   (1) Reset metric to that spot
  _g.reset(_xspot_center0);
  
  //   (2) Create Cartesianized differences
  //double dt,dx,dy,dz;
  double dx,dy,dz;
  double rc = _xspot_center0[1];
  double st = std::sin(_xspot_center0[2]);
  double ct = std::cos(_xspot_center0[2]);
  double sp = std::sin(_xspot_center0[3]);
  double cp = std::cos(_xspot_center0[3]);
  //dt = t - xspot_center[0];
  dx = r*std::sin(theta)*std::cos(phi) - rc*st*cp;
  dy = r*std::sin(theta)*std::sin(phi) - rc*st*sp;
  dz = r*std::cos(theta) - rc*ct;

 
  //   (3) Return to spherical coords at spot position
  FourVector<double> x(_g);
  x.mkcon(0.0, st*cp*dx+st*sp*dy+ct*dz, (ct*cp*dx + ct*sp*dy - st*dz)/rc, (sp*dx-cp*dy)/(rc*st));


  
  //   (3b) Get flow velocity
  FourVector<double> u(_g);
  u = _afv(_xspot_center0[0], _xspot_center0[1],
            _xspot_center0[2],_xspot_center0[3]);

  //   (4) Get differential radius;
  double r2 = ( (x*x) + std::pow( (x*u), 2) );

  //   (5) Reset metric to ray position
  _g.reset(t,r,theta,phi);

  return std::sqrt(r2);
}

}
