#include "ed_shearing_infall_spot_johannsen.h"

namespace VRT2 {

ED_shearing_infall_spot_johannsen::ED_shearing_infall_spot_johannsen( Johannsen& g, double density_scale, double rSpot, double ri, AFV_InflowJohannsen& afv, double infallRate, double subKep, double outer_radius, double t0, double r0, double theta0, double phi0)
  : _g(g), _density_scale(density_scale), _rspot(rSpot), _ri(ri), _afv(afv), _infallRate(infallRate), _subKep(subKep), _outer_radius(outer_radius), _xCenter(0.0,4), _xN(0.0,4), _xspot_center0(0.0,4), _toffset(0)
{


  if ( _infallRate <= 0 || _infallRate > 1)
  {
    std::cerr << "ed_shearing_infall_spot_johannsen.h assumes 0 < infall <= 1.\n";
    std::abort();
  }

  //Set spot center
  _xspot_center0[0] = t0;
  _xspot_center0[1] = r0;
  _xspot_center0[2] = theta0;
  _xspot_center0[3] = phi0;


  //Set table to not be generated yet
  _table_gen = false;
  

}

double ED_shearing_infall_spot_johannsen::get_density(const double t, const double r, const double theta, const double phi)
{
  if (!_table_gen)
  {
    std::cerr << "Spot table not generated exiting\n";
    std::abort();
  }
  //Check if off the table. Ensures you don't get nonsense
  if ( r < _xspot_tablesRev[1][0] || r > _xspot_tables[1][0] )
    return 1e-20;
  else
    return infall_density(t, r, theta, phi);
}



double ED_shearing_infall_spot_johannsen::infall_density(const double t, const double r, const double theta, const double phi)
{
  //Reset metric to spot position
  _g.reset(t,r,theta,phi);
  //det(g) of the metric at location
  const double detgNow = _g.detg();

  //Velocity factor of spot needed to get proper density
  FourVector<double> uNow = _afv(t,r,theta,phi);
  const double urN = uNow.con(1);

  //Amount of time spot has been evolving
  double tC0 = _xspot_center0[0] + _toffset;
  const double dt = (t-tC0);

  //Find location of spot in table and the expansion factor
  std::valarray<double> xN = curve_now(r);
  //Find the amount we need to rotate the stored integral curve to find the integral 
  //curve of the path
  double dphi = phi - xN[3];


  //Now find initial spot position by moving back dt units in time
  std::valarray<double> xspot0 = integral_curve(xN[0]-dt);

  //Apply shifts so we get on correct geodesic
  xspot0[0] = tC0;
  xspot0[2] = theta;
  xspot0[3] = xspot0[3] + dphi;

  //Find various expansion factors and detminant ratio for density
  //Reset metric to initial spot position
  _g.reset(xspot0);
  const double detgInit = _g.detg();
  const double detgRatio = detgInit/detgNow;
  FourVector<double> u0 = _afv(xspot0[0],xspot0[1],xspot0[2],xspot0[3]);
  const double ur0 = u0.con(1);
  
  //Spot Density expansion factor
  const double urRatio = std::fabs(ur0/urN);

 

  //Get the density
  double dr = delta_r(xspot0[0], xspot0[1], xspot0[2], xspot0[3]);
  double rho = detgRatio*_density_scale*
               FastMath::exp(-dr*dr/(2.*_rspot*_rspot))*urRatio;

  //Reset metric to ray position
  _g.reset(t,r,theta,phi);
  if ( dt < 0)
    rho *= FastMath::exp(-(dt*dt)/(2*0.1));

  return rho + 1e-20; 
  
}

double ED_shearing_infall_spot_johannsen::delta_r(const double t, const double r, const double theta, const double phi)
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
  //dt = t - _xspot_center0[0];
  dx = r*std::sin(theta)*std::cos(phi) - rc*st*cp;
  dy = r*std::sin(theta)*std::sin(phi) - rc*st*sp;
  dz = r*std::cos(theta) - rc*ct;

 
  //   (3) Return to spherical coords at spot position
  FourVector<double> x(_g);
  x.mkcon(0, st*cp*dx+st*sp*dy+ct*dz, (ct*cp*dx + ct*sp*dy - st*dz)/rc, (sp*dx-cp*dy)/(rc*st));


  
  //   (3b) Get flow velocity
  FourVector<double> u(_g);
  u = _afv(_xspot_center0[0], _xspot_center0[1],
            _xspot_center0[2],_xspot_center0[3]);

 
  //   (4) Get differential radius;
  double xu = x*u;
  double r2 = ( (x*x) + xu*xu );

  return std::sqrt(r2);
}


#define NDIM_ED_SOS 4
#define ED_SOS_TMAX 5000
#define MAXIT 1000000
void ED_shearing_infall_spot_johannsen::generate_spot_position_table()
{
  if (_xspot_center0[1]<=_g.horizon()) {
    std::cerr << "ED_SphericalOutflowingSpot::generate_position_table: Launch radius inside horizon!!!\n";
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
  y[0] = _xspot_center0[0];
  y[1] = _xspot_center0[1];
  y[2] = _xspot_center0[2];
  y[3] = _xspot_center0[3];

  h = hmax = -1.0e-1;
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
      std::cerr << "ED_SphericalOutflowingSpot::generate_position_table: Stepsize underflow\n";
      std::cerr << std::setw(15) << y[0]
                << std::setw(15) << y[1]
                << std::setw(15) << y[2]
                << std::setw(15) << y[3]
                << std::setw(15) << dydx[0]
                << std::setw(15) << dydx[1]
                << std::setw(15) << dydx[2]
                << std::setw(15) << dydx[3] << std::endl; 
      std::abort();
    }
  } while(y[1] < _outer_radius );

  //std::cerr << "Done integrating backwards\n";

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
  x = 0;

 
  y[0] = _xspot_center0[0];
  y[1] = _xspot_center0[1];
  y[2] = M_PI/2;
  y[3] = _xspot_center0[3];
  
  //std::cerr << "In here\n";
  

  //double rPrev = 0.0;
  
  h = hmax = hnext = 1.0e-1;
  hmin = 1.0e-8;
  do {
    derivs(x,y,dydx); // Initialize dydx
    //rPrev = y[1];
    get_yscal(h,x,y,dydx,yscal); // Initialize yscal

    rkqs(y,dydx,NDIM_ED_SOS,x,h,eps,yscal,hdid,hnext);

    h = std::min(hnext,hmax);
    /*
      std::cerr << std::setw(15) << y[0]
                << std::setw(15) << y[1]
                << std::setw(15) << y[2]
                << std::setw(15) << y[3]
                << std::setw(15) << y[4] << std::endl;
 */
    if (h<hmin) {
      std::cerr << "ED_shearing_final_spot::generate_position_table: Stepsize underflow\n";
      std::cerr << std::setw(15) << y[0]
                << std::setw(15) << y[1]
                << std::setw(15) << y[2]
                << std::setw(15) << y[3] << std::endl;
      std::abort();
    }
    // Save point
    for (int i=0; i<4; ++i)
      _xspot_tables[i].push_back(y[i]);

		if (_xspot_tables[0].size() == MAXIT){
			std::cerr << "xspot_table overflow! in ed_shearing_infall_spot_johannsen.cpp\n";
			std::cerr << std::setw(15) << y[0]
								<< std::setw(15) << y[1]
								<< std::setw(15) << y[2]
								<< std::setw(15) << y[3] 
								<< std::setw(15) << _g.horizon() << std::endl;
			std::exit(1);
		}
    
  }while( std::fabs(y[1] - _g.horizon())/_g.horizon() > 1e-5 );


  // Reverse order for lower interp (because vector doesn't have push_front?)
  for (int j=0; j<4; ++j)
  {
    std::vector<double> tmp = _xspot_tables[j];
    for (size_t i=0; i<tmp.size(); ++i)
      _xspot_tablesRev[j].push_back(tmp[tmp.size()-1-i]);
  }

}
#undef ED_SOS_TMAX

std::valarray<double>& ED_shearing_infall_spot_johannsen::curve_now(const double r)
{
  size_t i = _xspot_tablesRev[1].size()-1;
  double dr;

  if (r > _rArray[0] && r < _rArray[i]) {
    std::vector<double>::const_iterator p = std::lower_bound(_rArray.begin(), _rArray.end(), r);
    // p should now be an iterator to the first value less than x (special cases should already be seperated out!)
    i = p - _rArray.begin() - 1;
    dr = (r - _rArray[i])/(_rArray[i+1]-_rArray[i]);

    for (int j=0; j<4; ++j)
      _xN[j] = dr*_xspot_tablesRev[j][i+1] + (1.0-dr)*_xspot_tablesRev[j][i];

  }
  else if (r <=_xspot_tablesRev[1][0]){
    for (int j=0; j<4; ++j)
      _xN[j] = _xspot_tablesRev[j][0];
  }
  else{
    for (int j=0; j<4; ++j)
      _xN[j] = _xspot_tablesRev[j][i];
  }

  return _xN;  

}




std::valarray<double>& ED_shearing_infall_spot_johannsen::integral_curve(double t)
{
  size_t i = _xspot_tables[0].size()-1;
  double dt;
	double tN = t;
  if (tN > _tArray[0] && tN < _tArray[i]) {
    std::vector<double>::const_iterator p = std::lower_bound(_tArray.begin(), _tArray.end(), tN);
    // p should now be an iterator to the first value less than x (special cases should already be seperated out!)
    i = p -  _tArray.begin() - 1;
    dt = (tN - _tArray[i])/(_tArray[i+1]-_tArray[i]);

   for (int j=0; j<4; ++j)
      _xCenter[j] = dt*_xspot_tables[j][i+1] + (1.0-dt)*_xspot_tables[j][i];
  }
  else if (tN <=_xspot_tables[0][0]){
    for (int j=0; j<4; ++j)
      _xCenter[j] = _xspot_tables[j][0];
  }
  else{
    for (int j=0; j<4; ++j)
      _xCenter[j] = _xspot_tables[j][i];  
  }

  return _xCenter;  

}


/*** Reparametrization to Remove Singular behaviour near Horizon ***/
double ED_shearing_infall_spot_johannsen::reparametrize( const double y[])
{
  if (1.0/_g.ginv(0,0)<0.0) 
    return ( y[1]*y[1]*std::pow(std::min(1.0-_g.horizon()/y[1],std::fabs(1.0/_g.ginv(0,0))),1.0) );
  else
    return ( 1.0E300 );

}



/*** Derivatives for rkqs and rkck in propagate ***/
void ED_shearing_infall_spot_johannsen::derivs(double x, const double y[], double dydx[])
{
  // Reset stuff (x=lambda, y[0]=t, y[1]=r, y[2]=theta, y[3]=phi)
  _g.reset(y);
  double reparam = reparametrize(y);
  // Get velocity
  static FourVector<double> u(_g);
  u = _afv(y[0],y[1],y[2],y[3]);
        
  // Set dx/dt
  for (size_t i=0; i<4; ++i)
    dydx[i] = u.con(i)*reparam;

}




#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
int ED_shearing_infall_spot_johannsen::rkqs(double y[], double dydx[], int n, double& x, double htry, 
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
    htemp= ( vrt2_isinf(errmax) ? 0.5*h : SAFETY*h*FastMath::pow(errmax,PSHRNK) );
    htemp = (h >= 0.0 ? std::max(htemp,0.1*h) : std::min(htemp,0.1*h));
    xnew=x+htemp;
    if (xnew == x){
      return 1;
    }
    h = htemp;
  }

  if (errmax > ERRCON)
    hnext=SAFETY*h*FastMath::pow(errmax,PGROW);
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

void ED_shearing_infall_spot_johannsen::rkck(double y[], double dydx[], 
    int n, double x, double h, double yout[], double yerr[])
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

void ED_shearing_infall_spot_johannsen::generate_table()
{
  //Generate spot center integral curve in proper time
	for ( int i = 0; i < 4; i++)
	{
		_xspot_tables[i].resize(0);
		_xspot_tablesRev[i].resize(0);
	}
  generate_spot_position_table();
  for ( size_t j = 0; j < _xspot_tables[0].size(); ++j)
  {
    _tArray.push_back(_xspot_tables[0][j]);
    _rArray.push_back(_xspot_tablesRev[1][j]);
  }

	_table_gen = true;
}

void ED_shearing_infall_spot_johannsen::set_table(std::vector<double> spot_tables[4])
{
	for ( int i = 0; i < 4; ++i)
	{
		_xspot_tables[i].resize(0);
		_xspot_tablesRev[i].resize(0);
	}
  
  for ( int i = 0; i < 4; ++i)
    _xspot_tables[i] = spot_tables[i];

  // Reverse order for lower interp (because vector doesn't have push_front?)
  for (int j=0; j<4; ++j)
  {
    std::vector<double> tmp = _xspot_tables[j];
    for (size_t i=0; i<tmp.size(); ++i)
      _xspot_tablesRev[j].push_back(tmp[tmp.size()-1-i]);
  }
  for ( size_t j = 0; j < _xspot_tables[0].size(); ++j)
  {
    _tArray.push_back(_xspot_tables[0][j]);
    _rArray.push_back(_xspot_tablesRev[1][j]);
  }

	_table_gen = true;
}


void ED_shearing_infall_spot_johannsen::export_table(std::vector<double> spot_tables[4])
{
  for ( int i = 0; i < 4; i++)
    spot_tables[i] = _xspot_tables[i];
}





}
