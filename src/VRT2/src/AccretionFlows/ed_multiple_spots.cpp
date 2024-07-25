#include "ed_multiple_spots.h"
namespace VRT2 {

ED_multiple_spots::ED_multiple_spots( Metric& g, double ri, 
    AFV_ShearingInflow& afv, double infallRate, double subKep, 
    double outer_radius,
    std::vector<std::vector<double> > spot_parameters)
  : _g(g), _ri(ri), _afv(afv), _infallRate(infallRate), _subKep(subKep), _outer_radius(outer_radius), _xinitial(0.0, 4), _xnow(0.0, 4), _toffset(0)
{


  if ( _infallRate <= 0 || _infallRate > 1)
  {
    std::cerr << "ed_shearing_infall_spot.h assumes that 0 < alphaR <= 1.\n";
    std::abort();
  }

  if (spot_parameters[0].size() != 6)
  {
    std::cerr << "ed_multiple_spots.h needs 6 parameters per spot!.\n";
    std::abort();
  }

  //Copy the spot parameters over (notice row major order because C++)
  _spot_parameters.resize(spot_parameters.size());
  for ( size_t i = 0; i < spot_parameters.size(); ++i )
    for ( size_t j = 0; j < spot_parameters[0].size(); ++j)
      _spot_parameters[i].push_back(spot_parameters[i][j]);

  _table_gen = false;
  

}

double ED_multiple_spots::get_density(const double t, const double r, const double theta, const double phi)
{
  if (!_table_gen)
  {
    std::cerr << "Spot table not generated exiting\n";
    std::abort();
  }
  //Check if off the table. Ensures you don't get nonsense
  if ( r < _xspot_tablesRev[0][1] || r > _xspot_tables[0][1] )
    return 1e-20;
  else
    return infall_density(t, r, theta, phi);
}



double ED_multiple_spots::infall_density(const double t, const double r, const double theta, const double phi)
{
  double density = 1e-20;
  //Reset metric to spot position
  _g.reset(t,r,theta,phi);
  //det(g) of the metric at location
  const double detgNow = _g.detg();

  //Velocity factor of spot needed to get proper density
  _g.reset(t,r,M_PI/2.0,phi);
  FourVector<double> uNow = _afv(t,r,M_PI/2.0,phi);
  const double urN = uNow.con(1);
  std::valarray<double> xcenter(0.0, 4);
  //Now loop over the spots
  //#pragma omp simd
  for ( size_t i = 0; i < _spot_parameters.size(); ++i ){

    //Amount of time spot has been evolving
    double tC0 = _spot_parameters[i][2] + _toffset;
    double dt = (t-tC0);
    //Spot really isn't on so continue past this for speed!
    if ( dt < -10)
      break;

    if ( dt > 1000 )
      continue;

    //Find location of spot in table and the expansion factor and places it in _xnow
    curve_now(r, _xnow);
    //Find the amount we need to rotate the stored integral curve to find the integral 
    //curve of the path
    double dphi = phi - _xnow[3];

    if (_xnow[1] < 1.1*_g.horizon())
      continue;


    //Now find initial spot position by moving back dt units in time
    integral_curve(_xnow[0]-dt, _xinitial);

    //Apply shifts so we get on correct geodesic
    _xinitial[0] = tC0;
    _xinitial[2] = theta;
    _xinitial[3] = _xinitial[3] + dphi;

    //Find various expansion factors and determinant ratio for density
    //Reset metric to initial spot position
    _g.reset(_xinitial);
    const double detgInit = _g.detg(); //actually sqrt det g
    const double detgRatio = detgInit/detgNow;
    _g.reset(_xinitial[0],_xinitial[1],M_PI/2.0,_xinitial[3]);
    FourVector<double> u0 = _afv(_xinitial[0],_xinitial[1],M_PI/2.0,_xinitial[3]);
    const double ur0 = u0.con(1);
  
    //Spot expansion factor, simplifies greatly because no theta dependence
    const double utRatio = ur0/urN;
 
    //Don't worry we should be contiguous right?
    xcenter[0] = _spot_parameters[i][2];
    xcenter[1] = _spot_parameters[i][3];
    xcenter[2] = _spot_parameters[i][4];
    xcenter[3] = _spot_parameters[i][5];
    //Get the density
    double dr = delta_r(_xinitial, xcenter);
    double rspot2 = _spot_parameters[i][1]*_spot_parameters[i][1];
    double rho = detgRatio*_spot_parameters[i][0]*
                 FastMath::exp(-dr*dr/(2.*rspot2))*utRatio;
  
    //Smooth the turn on because RK4 issues...
    if ( dt < 0)
      rho *= FastMath::exp(-(dt*dt)/(2*0.1));
    
    //Now add the contribution to density.
    density += rho;
  }
  //Reset metric to ray position
  _g.reset(t,r,theta,phi);

  return density + 1e-20; 
  
}

inline double ED_multiple_spots::delta_r(const std::valarray<double>& x, const std::valarray<double>& xcenter)
{
  //   (1) Reset metric to that spot
  _g.reset(xcenter);
  
  //   (2) Create Cartesianized differences
  //double dt,dx,dy,dz;
  double dx,dy,dz;
  double rc = xcenter[1];
  double st = std::sin(xcenter[2]);
  double ct = std::cos(xcenter[2]);
  double sp = std::sin(xcenter[3]);
  double cp = std::cos(xcenter[3]);
  dx = x[1]*std::sin(x[2])*std::cos(x[3]) - rc*st*cp;
  dy = x[1]*std::sin(x[2])*std::sin(x[3]) - rc*st*sp;
  dz = x[1]*std::cos(x[2]) - rc*ct;

 
  //   (3) Return to spherical coords at spot position
  FourVector<double> xdiff(_g);
  xdiff.mkcon(0, st*cp*dx+st*sp*dy+ct*dz, (ct*cp*dx + ct*sp*dy - st*dz)/rc, (sp*dx-cp*dy)/(rc*st));


  
  //   (3b) Get flow velocity
  FourVector<double> u(_g);
  u = _afv(xcenter[0], xcenter[1],
           xcenter[2],xcenter[3]);

 
  //   (4) Get differential radius;
  double xu = xdiff*u;
  double r2 = ( (xdiff*xdiff) + xu*xu );

  return std::sqrt(r2);
}


#define NDIM_ED_SOS 4
#define ED_SOS_TMAX 5000
void ED_multiple_spots::generate_spot_position_table()
{

  // Zero tabls
  _xspot_tables.resize(0);

  // Define stepping stuff
  double eps = 1.0e-8;
  double h, hmin, hmax, hdid, hnext;

  // Define x, y, yscal, and dydx
  double x=0.0,y[NDIM_ED_SOS], yscal[NDIM_ED_SOS], dydx[NDIM_ED_SOS];
  

  // Integrate to horizon
  x = 0;

 
  y[0] = 0.0;
  y[1] = _outer_radius;
  y[2] = M_PI/2;
  y[3] = 0.0;
  
  
  h = hmax = hnext = 0.5;
  hmin = 1.0e-8;
  do {
    derivs(x,y,dydx); // Initialize dydx
    //rPrev = y[1];
    get_yscal(h,x,y,dydx,yscal); // Initialize yscal

    rkqs(y,dydx,NDIM_ED_SOS,x,h,eps,yscal,hdid,hnext);

    h = std::min(hnext,hmax);

    if (h<hmin) {
      std::cerr << "ED_shearing_final_spot::generate_position_table: Stepsize underflow\n";
      std::cerr << std::setw(15) << y[0]
                << std::setw(15) << y[1]
                << std::setw(15) << y[2]
                << std::setw(15) << y[3] << std::endl;
      std::abort();
    }
    // Save point
    _xspot_tables.push_back(std::vector<double>(y, y+sizeof(y)/sizeof(y[0])));

    
  }while( std::fabs(y[1] - _g.horizon())/_g.horizon() > 1e-3 );


  // Reverse order for lower interp (because vector doesn't have push_front?)
  for (size_t j=0; j< _xspot_tables.size(); ++j)
  {
    _xspot_tablesRev.push_back(_xspot_tables[_xspot_tables.size()-1-j]);
  }



}
#undef ED_SOS_TMAX

void ED_multiple_spots::curve_now(const double r, std::valarray<double>& xnow)
{
  size_t i = _xspot_tablesRev.size()-1;
  double dr;

  if (r > _rArray[0] && r < _rArray[i]) {
    std::vector<double>::const_iterator p = std::lower_bound(_rArray.begin(), _rArray.end(), r);
    // p should now be an iterator to the first value less than x (special cases should already be seperated out!)
    i = p - _rArray.begin() - 1;
    dr = (r - _rArray[i])/(_rArray[i+1]-_rArray[i]);

    for (int j=0; j<4; ++j)
      xnow[j] = dr*_xspot_tablesRev[i+1][j] + (1.0-dr)*_xspot_tablesRev[i][j];


  }
  else if (r <=_xspot_tablesRev[1][0]){
    for (int j=0; j<4; ++j)
      xnow[j] = _xspot_tablesRev[0][j];
  }
  else{
    for (int j=0; j<4; ++j)
      xnow[j] = _xspot_tablesRev[i][j];
  }

}




void ED_multiple_spots::integral_curve(double t, std::valarray<double>& xinitial)
{
  size_t i = _xspot_tables.size()-1;
  double dt;
	double tN = t;
  if (tN > _tArray[0] && tN < _tArray[i]) {
    std::vector<double>::const_iterator p = std::lower_bound(_tArray.begin(), _tArray.end(), tN);
    // p should now be an iterator to the first value less than x (special cases should already be seperated out!)
    i = p -  _tArray.begin() - 1;
    dt = (tN - _tArray[i])/(_tArray[i+1]-_tArray[i]);

   for (int j=0; j<4; ++j)
      xinitial[j] = dt*_xspot_tables[i+1][j] + (1.0-dt)*_xspot_tables[i][j];
  }
  else if (tN <=_xspot_tables[0][0]){
    for (int j=0; j<4; ++j)
      xinitial[j] = _xspot_tables[0][j];
  }
  else{
    for (int j=0; j<4; ++j)
      xinitial[j] = _xspot_tables[i][j];
  }

}

/*** Reparametrization to Remove Singular behaviour near Horizon ***/
double ED_multiple_spots::reparametrize( const double y[])
{
  if (1.0/_g.ginv(0,0)<0.0) 
    return ( y[1]*y[1]*std::pow(std::min(1.0-_g.horizon()/y[1],std::fabs(1.0/_g.ginv(0,0))),1.0) );
  else
    return ( 1.0E300 );

}

/*** Derivatives for rkqs and rkck in propagate ***/
void ED_multiple_spots::derivs(double x, const double y[], double dydx[])
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
int ED_multiple_spots::rkqs(double y[], double dydx[], int n, double& x, double htry, 
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

void ED_multiple_spots::rkck(double y[], double dydx[], 
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

void ED_multiple_spots::generate_table()
{
  //Generate spot center integral curve in proper time
  _xspot_tables.resize(0);
  _xspot_tablesRev.resize(0);
  generate_spot_position_table();
  for ( size_t j = 0; j < _xspot_tables.size(); ++j){
    _tArray.push_back(_xspot_tables[j][0]);
    _rArray.push_back(_xspot_tablesRev[j][1]);
  }
  _table_gen = true;
}

void ED_multiple_spots::set_table(std::vector<std::vector<double> > spot_tables)
{
  _xspot_tables.resize(0);
  _xspot_tablesRev.resize(0);
  
  for ( size_t i = 0; i < spot_tables.size(); ++i)
    _xspot_tables.push_back(spot_tables[i]);

  // Reverse order for lower interp (because vector doesn't have push_front?)
  for (size_t j=0; j<spot_tables.size(); ++j)
  {
    _xspot_tablesRev.push_back(spot_tables[spot_tables.size()-1-j]);
  }
  for ( size_t j = 0; j < _xspot_tables[0].size(); ++j)
  {
    _tArray.push_back(_xspot_tables[j][0]);
    _rArray.push_back(_xspot_tablesRev[j][1]);
  }
  _table_gen = true;
}


void ED_multiple_spots::export_table(std::vector<std::vector<double> >& spot_tables)
{
  spot_tables.resize(0);
  for ( size_t i = 0; i < _xspot_tables.size(); i++)
    spot_tables.push_back(_xspot_tables[i]);
}





}
