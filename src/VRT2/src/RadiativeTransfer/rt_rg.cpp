// Include Statements
#include "rt_rg.h"
#include <mpi.h>


#define N_INTS 4
#define TINY 1.0e-10

namespace VRT2 {
RT_RungeKutta::RT_RungeKutta(Metric& g)
  : RadiativeTransfer(g)
{
}

RT_RungeKutta::RT_RungeKutta(const double y[], Metric& g)
  : RadiativeTransfer(g)
{
}

RT_RungeKutta::RT_RungeKutta(FourVector<double>& x, FourVector<double>& k, Metric& g)
  : RadiativeTransfer(g)
{
}

// Integration of Stokes parameters
// Need to be given a vector of y and dydx (points along rays).
// Returns (I,Q,U,V)
// Assumes that lambda is decreasing towards the observer
std::valarray<double>& RT_RungeKutta::IQUV_integrate(std::vector<double> ya[], std::vector<double> dydxa[], std::valarray<double>& iquv)
{
  _output_flag = false;

  // If only one element in ray, don't bother integrating
  size_t N = ya[0].size();
  if (N<=1)
    return iquv;
  std::valarray<double> iquv_start = iquv;

  // Set pointers to these arrays
  _ya = ya;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (size_t i=0; i<N; ++i)
    _ya[0][i] *= -1.0;
  _dydxa = dydxa;
  static double y[8], dydx[8];



  // Do any special initialization that needs to be done
  IQUV_integrate_initialize(ya,dydxa,iquv);



  // TEMPORARY SPPED UP FOR WHEN THERE IS NO STUFF IN THE WAY
  //if (iquv[0] == 0)
  //  return iquv;


  ///////
  // TMP TO OUTPUT ALL THE RT STUFF ALONG THE RAY
  //
  //dump_ray("ray.dump");

  

  // Set up arrays for integration
  //double tiny = TINY; // accuracy for yscale
  static double iquv_int[N_INTS], diquv[N_INTS], iquv_scal[N_INTS];
  iquv_int[0] = iquv_start[0];
  iquv_int[1] = iquv_start[1];
  iquv_int[2] = iquv_start[2];
  iquv_int[3] = iquv_start[3];

  // Integration step details (hmax is 4 ray steps)
  double lambda, h, hdid=0, hnext=0, hmax = (-4*_ya[0][N-1]/N);

  // Accuracy level
  const double eps = 1.0e-6;
  double yscal_const;

  // DEBUG HACK, TRAPAZOIDAL RULE TO GET INTEGRAL
  // Get absolute scale by rough approximation
  for (size_t i=N-1; i>=1; --i) {
    derivs(0.5*(_ya[0][i-1]+_ya[0][i]),iquv_int,diquv);
    iquv_int[0] += diquv[0]*(_ya[0][i-1]-_ya[0][i]);
  }
  yscal_const = iquv_int[0];
  //std::cout << std::setw(15) << yscal_const << std::endl;
  iquv_int[0] = iquv_start[0];
  iquv_int[1] = iquv_start[1];
  iquv_int[2] = iquv_start[2];
  iquv_int[3] = iquv_start[3];

  // Accuracy Check & Rotation Args
  static double y0[8], dydx0[8];
  double lambda0, iquv_int0[N_INTS];
  bool bad_step;

  // Initialize integration
  lambda = _ya[0][N-1]; // start at last point
  interp(lambda,y,dydx);
  _g.reset(y);
  reinitialize(y);

  h = -stable_step_size(-hmax,y,dydx);


  /*** Perform Integration ***/
  for (lambda=_ya[0][N-1]; lambda>_ya[0][0];){

    // Make sure that we stay lambda>=0
    if ( (lambda + h) < _ya[0][0] )
      h = _ya[0][0] - lambda;

    /** Take a Stable Step **/
    // Save old values in case we must readjust step size for stability
    bad_step = false;
    lambda0 = lambda;
    interp(lambda0,y0,dydx0);
    //#pragma omp simd
    for (int i=0; i<N_INTS; ++i)
      iquv_int0[i] = iquv_int[i];

    // Loop to ensure stability
    int n_bad_stps=0;
    do {
      ++n_bad_stps;
      if (n_bad_stps>1000)
	std::cerr << "Too many steps!\n";

      if (bad_step) {
	// Reset to starting position
	lambda = lambda0;
	//#pragma omp simd
	for (int i=0; i<N_INTS; ++i)
	  iquv_int[i] = iquv_int0[i];
      }

      // Call derivs first (Note that reinitialization is done in derivs!)
      derivs(lambda,iquv_int,diquv);

      // Get iquv_scal
      //#pragma omp simd
      for (int i=0;i<N_INTS;++i)
	iquv_scal[i] = std::fabs(1.0*iquv_int[i]) + std::fabs(h*diquv[i]) + yscal_const;

      /*
      if (vrt2_isnan(iquv_int[0]))
	std::cout << std::setw(15) << lambda
		  << std::setw(15) << h
		  << std::setw(15) << y[0]
		  << std::setw(15) << y[1]
		  << std::setw(15) << y[2]
		  << std::setw(15) << y[3]
		  << std::setw(15) << dydx[0]
		  << std::setw(15) << dydx[1]
		  << std::setw(15) << dydx[2]
		  << std::setw(15) << dydx[3]
		  << std::setw(15) << diquv[0]
		  << std::setw(15) << diquv[1]
		  << std::setw(15) << diquv[2]
		  << std::setw(15) << diquv[3]
		  << std::setw(15) << iquv_scal[0]
		  << std::setw(15) << iquv_scal[1]
		  << std::setw(15) << iquv_scal[2]
		  << std::setw(15) << iquv_scal[3]
		  << std::endl;
      */

      // Call rkqs
      /*
      std::cout << std::setw(15) << lambda
		<< std::setw(15) << h
		<< std::setw(15) << y[0]
		<< std::setw(15) << y[1]
		<< std::setw(15) << y[2]
		<< std::setw(15) << y[3];
      */


      rkqs(iquv_int,diquv,N_INTS,lambda,h,eps,iquv_scal,hdid,hnext);

      // Interpolate and reinitialize at new position
      interp(lambda,y,dydx);
      _g.reset(y);
      reinitialize(y);

      // Get the stable step size at the new position
      h = -stable_step_size(-hdid,y,dydx);

      // Check stability of step
      if (h!=hdid || vrt2_isnan(iquv_int[0]) || vrt2_isnan(iquv_int[1]) || vrt2_isnan(iquv_int[2]) || vrt2_isnan(iquv_int[3]))
	bad_step = true;
      else
	bad_step = false;
    }
    while (bad_step);

    // Get next stable step size
    h = std::max(hnext,hmax); // Remember h is negative now (inverted lambda array to make sorted)

    // Do linear change step
    IQUV_rotate(iquv_int,lambda0,y0,dydx0,lambda,y,dydx);

  }


  //std::cout << "\n\n";

  // Save into return values
  for (int i=0;i<4;++i)
    iquv[i] = iquv_int[i];

 
  if (std::fabs(iquv[1])>iquv[0] || std::fabs(iquv[2])>iquv[0] || std::fabs(iquv[3])>iquv[0] || vrt2_isnan(iquv[0]) || vrt2_isnan(iquv[1]) || vrt2_isnan(iquv[2]) || vrt2_isnan(iquv[3]))
    //if (vrt2_isnan(iquv[0]) || vrt2_isnan(iquv[1]) || vrt2_isnan(iquv[2]) || vrt2_isnan(iquv[3]))
    std::cout << "NaN's found in RT at end:"
	      << std::setw(15) << iquv[0]
	      << std::setw(15) << iquv[1]
	      << std::setw(15) << iquv[2]
	      << std::setw(15) << iquv[3]
	      << std::endl;

  // Output if there has been a mistake
  if (std::fabs(iquv[1]) > iquv[0]
      || std::fabs(iquv[2]) > iquv[0]
      || std::fabs(iquv[3]) > iquv[0]
      || vrt2_isnan(iquv[0])
      || vrt2_isnan(iquv[1])
      || vrt2_isnan(iquv[2])
      || vrt2_isnan(iquv[3]) ) {
    std::cout.setf(std::ios::scientific);
    std::cout << std::setprecision(5);
    std::cout << "Radiative Transfer Error:\n"
	      << "Istart: "
	      << std::setw(15) << iquv_start[0]
	      << std::setw(15) << iquv_start[1]
	      << std::setw(15) << iquv_start[2]
	      << std::setw(15) << iquv_start[3]
	      << "\nIinteg: "
	      << std::setw(15) << iquv[0]
	      << std::setw(15) << iquv[1]
	      << std::setw(15) << iquv[2]
	      << std::setw(15) << iquv[3]
	      << "\n\n";
  }

  return iquv;
}

// Determination of step size which satisfies the stability criterion
double RT_RungeKutta::stable_step_size(double h, const double y[], const double dydx[])
{
  return h;
}


// Derivs
void RT_RungeKutta::derivs(double lambda, const double iquv[], double diquv[])
{
  static double y[8], dydx[8];
  // Linearly interpolate
  interp(lambda,y,dydx);
    
  // Initialize to current point
  _g.reset(y);
  reinitialize(y);

  std::valarray<double> iquv_ems = IQUV_ems(dydx);
  std::valarray<double> iquv_abs = IQUV_abs(iquv,dydx);

  for (int i=0; i<4; ++i)
    diquv[i] = -(iquv_ems[i] - iquv_abs[i]); // since inverted lambda decreases in forward time now


  bool bad = vrt2_isnan(diquv[0]);
  for (size_t i=1; i<5; i++)
    bad = bad || vrt2_isnan(diquv[i]);
  bool prev_good = vrt2_isnan(iquv[0]);
  for (size_t i=1; i<5; i++)
    prev_good = prev_good || !(vrt2_isnan(iquv[i]));
  if ((bad && prev_good) || _output_flag )
    std::cout << "Failed in RT_RungeKutta:"
	      << std::setw(15) << lambda
	      << std::setw(15) << iquv[0]
	      << std::setw(15) << iquv[1]
	      << std::setw(15) << iquv[2]
	      << std::setw(15) << iquv[3]
      //<< std::setw(15) << iquv[4]
	      << std::setw(15) << diquv[0]
	      << std::setw(15) << diquv[1]
	      << std::setw(15) << diquv[2]
	      << std::setw(15) << diquv[3]
      //<< std::setw(15) << diquv[4]
	      << std::setw(15) << iquv_ems[0]
	      << std::setw(15) << iquv_ems[1]
	      << std::setw(15) << iquv_ems[2]
	      << std::setw(15) << iquv_ems[3]
	      << std::setw(15) << iquv_abs[0]
	      << std::setw(15) << iquv_abs[1]
	      << std::setw(15) << iquv_abs[2]
	      << std::setw(15) << iquv_abs[3]
	      << std::setw(15) << y[0]
	      << std::setw(15) << y[1]
	      << std::setw(15) << y[2]
	      << std::setw(15) << y[3]
	      << std::endl;
}

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
int RT_RungeKutta::rkqs(double y[], double dydx[], int n, double& x, double htry, double eps, double yscal[], double& hdid, double& hnext)
{
  int i;
  double errmax,h,htemp,xnew,yerr[N_INTS],ytemp[N_INTS]; 
  h=htry;
  for (;;) {
    rkck(y,dydx,n,x,h,ytemp,yerr);
    errmax=0.0;
    for (i=0;i<n;i++){
      if (yscal[i]>0.0 && std::fabs(yerr[i]/yscal[i])>errmax)
	errmax = std::fabs(yerr[i]/yscal[i]);
    }
    errmax /= eps;
    if (errmax <= 1.0)
      break;
    htemp=SAFETY*h*std::pow(errmax,PSHRNK);
    htemp=(h >= 0.0 ? std::max(htemp,0.1*h) : std::min(htemp,0.1*h));
    xnew=x+htemp;

    if (std::fabs(xnew-x)/std::fabs(xnew+x) <1e-10){
      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD ,&world_rank);
      std::cerr << "stepsize underflow in rkqs" << std::endl;
      std::cout << "rkqs: "
		<< std::setw(15) << world_rank 
		<< std::setw(15) << htemp
		<< std::setw(15) << xnew
		<< std::setw(15) << x
		<< std::setw(15) << htry
		<< std::setw(15) << yerr[0]
		<< std::setw(15) << yerr[1]
		<< std::setw(15) << yerr[2]
		<< std::setw(15) << yerr[3]
		<< std::setw(15) << yscal[0]
		<< std::setw(15) << yscal[1]
		<< std::setw(15) << yscal[2]
		<< std::setw(15) << yscal[3]
		<< std::setw(15) << yerr[0]/yscal[0]
		<< std::setw(15) << yerr[1]/yscal[1]
		<< std::setw(15) << yerr[2]/yscal[2]
		<< std::setw(15) << yerr[3]/yscal[3]
		<< std::setw(15) << eps
		<< "\n\n" << std::endl;

      dump_ray("ray.dump");
      _output_flag = true;
      return 1;
    }
    h = htemp;
  }
  if (errmax > ERRCON) 
    hnext=SAFETY*h*std::pow(errmax,PGROW);
  else
    hnext=5.0*h;
  x += (hdid=h);
  //#pragma omp simd
  for (i=0;i<n;i++)
    y[i]=ytemp[i];
  return 0;
}
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON

void RT_RungeKutta::rkck(double y[], double dydx[], int n, double x, double h, double yout[], double yerr[])
{
  int i;
  static constexpr double a2=0.2,a3=0.3,a4=0.6,a5=1.0,a6=0.875,b21=0.2,
    b31=3.0/40.0,b32=9.0/40.0,b41=0.3,b42 = -0.9,b43=1.2,
    b51 = -11.0/54.0, b52=2.5,b53 = -70.0/27.0,b54=35.0/27.0,
    b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
    b64=44275.0/110592.0,b65=253.0/4096.0,c1=37.0/378.0,
    c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
    dc5 = -277.00/14336.0;
  static constexpr double dc1=c1-2825.0/27648.0,dc3=c3-18575.0/48384.0,
    dc4=c4-13525.0/55296.0,dc6=c6-0.25;
  static double ak2[N_INTS],ak3[N_INTS],ak4[N_INTS],ak5[N_INTS],ak6[N_INTS],ytemp[N_INTS];

  //#pragma omp simd
  for (i=0;i<N_INTS;i++)
    ytemp[i]=y[i]+b21*h*dydx[i];
  derivs(x+a2*h,ytemp,ak2);
  //#pragma omp simd
  for (i=0;i<N_INTS;i++)
    ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i]);
  derivs(x+a3*h,ytemp,ak3);
  //#pragma omp simd
  for (i=0;i<N_INTS;i++)
    ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i]);
  derivs(x+a4*h,ytemp,ak4);
  //#pragma omp simd
  for (i=0;i<N_INTS;i++)
    ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
  derivs(x+a5*h,ytemp,ak5);
  //#pragma omp simd
  for (i=0;i<N_INTS;i++)
    ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
  derivs(x+a6*h,ytemp,ak6);
  //#pragma omp simd
  for (i=0;i<N_INTS;i++)
    yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
  //#pragma omp simd
  for (i=0;i<N_INTS;i++)
    yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);
}
#undef TINY
#undef N_INTS



};
