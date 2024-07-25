#include "ed_spherical_orbiting_spot.h"

namespace VRT2 {
ED_SphericalOrbitingSpot::ED_SphericalOrbitingSpot(Metric& g, double density_scale, double rspot, AFV_ShearingInflow& afv, double t0, double r0, double theta0, double phi0, double tstart, double tend)
  : _g(g), _density_scale(density_scale), _rspot(rspot), _afv(afv), _xspot_center(0.0,4)
{
 generate_spot_position_table(t0,r0,theta0,phi0,tstart,tend);
}

void ED_SphericalOrbitingSpot::reset(double t0, double r0, double theta0, double phi0,double tstart, double tend)
{
  generate_spot_position_table(t0,r0,theta0,phi0, tstart, tend);
}

double ED_SphericalOrbitingSpot::get_density(double t, double r, double theta, double phi)
{
  // (A) Test to make sure that time has started
  if ( t>=_xspot_tables[0][0] || t <= _xspot_tables[0].back() ) 
  {
    // (B) Get differential radius from spot center
    _xspot_center = spot_center(t - _toffset);    
    double dr = delta_r(t,r,theta,phi,_xspot_center);
    double dt = t-_t0+_toffset;
    /*
    std::cerr << "Spot center:"
	      << std::setw(15) << _xspot_center[0]
	      << std::setw(15) << _xspot_center[1]
	      << std::setw(15) << _xspot_center[2]
	      << std::setw(15) << _xspot_center[3]
	      << std::setw(15) << dr
	      << std::setw(15) << _density_scale
	      << std::setw(15) << _rspot
	      << '\n';
    */

    // (C) Get density
    double rho = _density_scale * FastMath::exp(-dr*dr/(2.0*_rspot*_rspot)) + 1e-20;
    //Check if spot is on
    
 
    if ( dt >= 0)
      return rho;
    else
      return rho*FastMath::exp(-(t-_t0)*(t-_t0)/(2*1e-3));
  }
  return 1e-20;  
  
}

inline double ED_SphericalOrbitingSpot::delta_r(double t, double r, double theta, double phi, std::valarray<double> xspot_center)//,FourVector<double>& xspot_center)
{
  //   (1) Reset metric to that spot
  _g.reset(xspot_center);
  
  //   (2) Create Cartesianized differences
  //double dt,dx,dy,dz;
  double dx,dy,dz;
  double rc = xspot_center[1];
  double st = std::sin(xspot_center[2]);
  double ct = std::cos(xspot_center[2]);
  double sp = std::sin(xspot_center[3]);
  double cp = std::cos(xspot_center[3]);
  //dt = t - xspot_center[0];
  dx = r*std::sin(theta)*std::cos(phi) - rc*st*cp;
  dy = r*std::sin(theta)*std::sin(phi) - rc*st*sp;
  dz = r*std::cos(theta) - rc*ct;
  
  //   (3) Return to spherical coords at spot position
  FourVector<double> x(_g);
  x.mkcon(0.0, st*cp*dx+st*sp*dy+ct*dz, (ct*cp*dx + ct*sp*dy - st*dz)/rc, (sp*dx-cp*dy)/(rc*st));
  
  //   (3b) Get flow velocity
  FourVector<double> u(_g);
  u = _afv(xspot_center[0],xspot_center[1],xspot_center[2],xspot_center[3]);
  double ux = (x*u);
  //   (4) Get differential radius;
  double r2 = ( (x*x) + ux*ux );

  //   (5) Reset metric to ray position
  _g.reset(t,r,theta,phi);

  return std::sqrt(r2);
}

std::valarray<double>& ED_SphericalOrbitingSpot::spot_center(double t)
{
  size_t i = _xspot_tables[0].size()-1;
  double dt;
  _xspot_center[0] = t;
  if (t >_xspot_tables[0][0] && t <_xspot_tables[0][i]) {
    std::vector<double>::const_iterator p = std::lower_bound(_xspot_tables[0].begin(),_xspot_tables[0].end(),t);
    // p should now be an iterator to the first value less than x (special cases;tth should already be seperated out!)
    i = p - _xspot_tables[0].begin() - 1;
    dt = (t -_xspot_tables[0][i])/(_xspot_tables[0][i+1]-_xspot_tables[0][i]);
    for (int j=1; j<4; ++j)
      _xspot_center[j] = dt*_xspot_tables[j][i+1] + (1.0-dt)*_xspot_tables[j][i];
  }
  else if (t <=_xspot_tables[0][0])
    for (int j=1; j<4; ++j)
      _xspot_center[j] = _xspot_tables[j][0];
  else
    for (int j=1; j<4; ++j)
      _xspot_center[j] = _xspot_tables[j][i];

  return _xspot_center;  
}


void ED_SphericalOrbitingSpot::output_spot_path(std::string fname)
{
  std::ofstream out(fname);
  for (size_t i=0; i<_xspot_tables[0].size()-1; ++i)
  {
    _g.reset(_xspot_tables[0][i],_xspot_tables[1][i],_xspot_tables[2][i],_xspot_tables[3][i]);
    FourVector<double> u = _afv(_xspot_tables[0][i],_xspot_tables[1][i],_xspot_tables[2][i],_xspot_tables[3][i]);
    out << std::setw(15) << _xspot_tables[0][i]
	      << std::setw(15) << _xspot_tables[1][i]
	      << std::setw(15) << _xspot_tables[2][i]
	      << std::setw(15) << _xspot_tables[3][i]
	      << std::setw(15) << u.con(0)
	      << std::setw(15) << u.con(1)/u.con(0)
	      << std::setw(15) << u.con(2)/u.con(0)
	      << std::setw(15) << u.con(3)/u.con(0)
	      << std::endl;

  }
}


#define NDIM_ED_SOS 4
#define ED_SOS_TMAX 5000
void ED_SphericalOrbitingSpot::generate_spot_position_table(double t, double r, double theta, double phi, double tstart, double tend)
{
  // Save launch position
  _t0 = t;
  _r0 = r;
  _theta0 = theta;
  _phi0 = phi;

  if (_r0<=_g.horizon()) {
    std::cerr << "ED_SphericalOrbitingSpot::generate_position_table: Launch radius inside horizon!!!\n";
    std::abort();
  }

  // Zero tables
  for (size_t i=0; i<4; ++i)
    _xspot_tables[i].resize(0);

  // Define stepping stuff
  double eps = 1.0e-10;
  double h, hmin, hmax, hdid, hnext;

  // Define x, y, yscal, and dydx
  double x=0.0,y[NDIM_ED_SOS], yscal[NDIM_ED_SOS], dydx[NDIM_ED_SOS];
  
  // Integrate backwards to tstart
  // Initialize y
  y[0] = _t0;
  y[1] = _r0;
  y[2] = _theta0;
  y[3] = _phi0;

  h = hmax = hnext = -1.0e-1;
  hmin = -1.0e-8;
  do {
    derivs(x,y,dydx); // Initialize dydx

    // Save point
    for (int i=0; i<4; ++i)
      _xspot_tables[i].push_back(y[i]);

    get_yscal(h,x,y,dydx,yscal); // Initialize yscal

    rkqs(y,dydx,NDIM_ED_SOS,x,h,eps,yscal,hdid,hnext);

    h = std::max(hnext,hmax);

    if (h>hmin) {
      std::cerr << "ED_SphericalOutflowingSpot::generate_position_table: Stepsize underflow\n"
                << std::setw(15) << t
                << std::setw(15) << r
                << std::setw(15) << theta
                << std::setw(15) << phi << std::endl;
      std::abort();
    }
  } while(y[0]>tstart);

  // Save point
  for (int i=0; i<4; ++i)
    _xspot_tables[i].push_back(y[i]);

  // Reverse order (because vector doesn't have push_front?)
  for (int j=0; j<4; ++j)
  {
    std::vector<double> tmp = _xspot_tables[j];
    for (size_t i=0; i<tmp.size(); ++i)
      _xspot_tables[j][i] = tmp[tmp.size()-1-i];
  }

  
  // Integrate forwards to tend
  // Initialize y
  y[0] = _t0;
  y[1] = _r0;
  y[2] = _theta0;
  y[3] = _phi0;

 
  h = hmax = hnext = 1.0e-1;
  hmin = 1.0e-8;
  
  do {
    derivs(x,y,dydx); // Initialize dydx

    get_yscal(h,x,y,dydx,yscal); // Initialize yscal

    rkqs(y,dydx,NDIM_ED_SOS,x,h,eps,yscal,hdid,hnext);

    h = std::min(hnext,hmax);

    if (h<hmin) {
      std::cerr << "ED_SphericalOrbitingSpot::generate_position_table: Stepsize underflow\n"
                << std::setw(15) << t
                << std::setw(15) << r
                << std::setw(15) << theta
                << std::setw(15) << phi << std::endl;
 
      std::abort();
    }

    // Save point
    for (int i=0; i<4; ++i)
      _xspot_tables[i].push_back(y[i]);

  } while(y[0]<tend && (y[1]-_g.horizon()) > 1e-5);
}
#undef ED_SOS_TMAX



/*** Derivatives for rkqs and rkck in propagate ***/
void ED_SphericalOrbitingSpot::derivs(double x, const double y[], double dydx[])
{
  // Reset stuff (x=lambda, y[0]=t, y[1]=r, y[2]=theta, y[3]=phi)
  _g.reset(y);

  // Get velocity
  static FourVector<double> u(_g);
  u = _afv(y[0],y[1],y[2],y[3]);

  // Set dx/dtau
  double reparam = reparameterize(y);
  for (size_t i=0; i<4; ++i)
    dydx[i] = u.con(i)*reparam;
}

/*** Reparametrization to Remove Singular behaviour near Horizon ***/
double ED_SphericalOrbitingSpot::reparameterize(const double y[])
{
  if (1.0/_g.ginv(0,0)<0.0) 
    return ( y[1]*y[1]*std::pow(std::min(1.0-_g.horizon()/y[1],std::fabs(1.0/_g.ginv(0,0))),1.0) );
  else
    return ( 1.0 );

}





#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
int ED_SphericalOrbitingSpot::rkqs(double y[], double dydx[], int n, double& x, double htry, 
				     double eps, double yscal[], double& hdid, double& hnext)
{
  static int i;
  static double errmax,h,htemp,xnew,yerr[NDIM_ED_SOS],ytemp[NDIM_ED_SOS]; 
  
  h=htry;
  for (;;) {
    rkck(y,dydx,n,x,h,ytemp,yerr);
    errmax=0.0;
    for (i=0;i<n;i++){
      if (std::fabs(yerr[i]/yscal[i])>errmax)
	errmax = std::fabs(yerr[i]/yscal[i]);
      if (vrt2_isnan(ytemp[i]))
	errmax = 10.0*eps;
    }
    errmax /= eps;

    if (errmax <= 1.0)
      break;
    htemp= ( vrt2_isinf(errmax) ? 0.5*h : SAFETY*h*std::pow(errmax,PSHRNK) );
    htemp = (h >= 0.0 ? std::max(htemp,0.1*h) : std::min(htemp,0.1*h));
    xnew=x+htemp;
    if (xnew == x){
      return 1;
    }
    h = htemp;
  }

  if (errmax > ERRCON)
    hnext=SAFETY*h*std::pow(errmax,PGROW);
  else
    hnext=5.0*h;
  x += (hdid=h);

  for (i=0;i<n;i++)
    y[i]=ytemp[i];

  return 0;
}
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON

void ED_SphericalOrbitingSpot::rkck(double y[], double dydx[], int n, double x, double h, 
				      double yout[], double yerr[])
{
  int i;
  static double a2=0.2,a3=0.3,a4=0.6,a5=1.0,a6=0.875,b21=0.2,
    b31=3.0/40.0,b32=9.0/40.0,b41=0.3,b42 = -0.9,b43=1.2,
    b51 = -11.0/54.0, b52=2.5,b53 = -70.0/27.0,b54=35.0/27.0,
    b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
    b64=44275.0/110592.0,b65=253.0/4096.0,c1=37.0/378.0,
    c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
    dc5 = -277.00/14336.0;
  static double dc1=c1-2825.0/27648.0,dc3=c3-18575.0/48384.0,
    dc4=c4-13525.0/55296.0,dc6=c6-0.25;
  static double ak2[NDIM_ED_SOS],ak3[NDIM_ED_SOS],ak4[NDIM_ED_SOS],
    ak5[NDIM_ED_SOS],ak6[NDIM_ED_SOS],ytemp[NDIM_ED_SOS];

  for (i=0;i<n;i++)
    ytemp[i]=y[i]+b21*h*dydx[i];
  derivs(x+a2*h,ytemp,ak2);
  for (i=0;i<n;i++)
    ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i]);
  derivs(x+a3*h,ytemp,ak3);
  for (i=0;i<n;i++)
    ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i]);
  derivs(x+a4*h,ytemp,ak4);
  for (i=0;i<n;i++)
    ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
  derivs(x+a5*h,ytemp,ak5);
  for (i=0;i<n;i++)
    ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
  derivs(x+a6*h,ytemp,ak6);
  for (i=0;i<n;i++)
    yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
  for (i=0;i<n;i++)
    yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);
}
#undef NDIM_ED_SOS

};
