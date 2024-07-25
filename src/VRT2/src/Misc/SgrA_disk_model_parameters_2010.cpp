#include "SgrA_disk_model_parameters_2010.h"

namespace VRT2 {
SgrA_PolintDiskModelParameters2010::SgrA_PolintDiskModelParameters2010(std::string fname, int aorder, int thetaorder, double a, double THETA)
  : _a(a), _THETA(THETA), _aorder(aorder), _thetaorder(thetaorder)
{
#ifdef MPI_MAP
  int rank = MPI::COMM_WORLD.Get_rank();
#else
  int rank = 0;
#endif


  a_.resize(0);
  theta_.resize(0);
  ne_.resize(0);
  Te_.resize(0);
  nnth_.resize(0);


  // Read data from file into tables
  if (rank==0)
  {
    std::ifstream in(fname.c_str());
    if (!in.is_open())
    {
      std::cerr << "Couldn't open " << fname << std::endl;
      std::exit(1);
    }

    in.ignore(4096,'\n');
    in.ignore(4096,'\n');
    do {
      double itmp;
      double tmp;
      in >> itmp;
      if (!in.eof())
      {
	in >> itmp;
	in >> tmp;
	a_.push_back(tmp);
	in >> tmp;
	theta_.push_back(tmp);
	in >> tmp;
	ne_.push_back(tmp);
	in >> tmp;
	Te_.push_back(tmp);
	in >> tmp;
	nnth_.push_back(tmp);
	in.ignore(4096,'\n');
      }
    } while (!in.eof());
  }
#ifdef MPI_MAP
  int tablesize;
  if (rank==0)
    tablesize=a_.size();
  MPI::COMM_WORLD.Bcast(&tablesize,1,MPI::INT,0);
  double *buff = new double[5*tablesize];
  if (rank==0)
    for (int i=0; i<tablesize; i++)
    {
      buff[5*i+0] = a_[i];
      buff[5*i+1] = theta_[i];
      buff[5*i+2] = ne_[i];
      buff[5*i+3] = Te_[i];
      buff[5*i+4] = nnth_[i];
    }
  MPI::COMM_WORLD.Bcast(&buff[0],5*tablesize,MPI::DOUBLE,0);
  if (rank!=0)
    for (int i=0; i<tablesize; i++)
    {
      a_.push_back(buff[5*i+0]);
      theta_.push_back(buff[5*i+1]);
      ne_.push_back(buff[5*i+2]);
      Te_.push_back(buff[5*i+3]);
      nnth_.push_back(buff[5*i+4]);
    }
  delete[] buff;
#endif

  // Get table limits (assumes that a & theta are listed in ascending order)
  amin_ = amax_ = a_[0];
  thetamin_ = thetamax_ = theta_[0];
  Na_ = 1;
  Ntheta_ = 1;
  for (size_t i=1; i<a_.size(); ++i) {
    if (a_[i]>amax_) {
      ++Na_;
      amax_ = a_[i];
    }
    if (theta_[i]>thetamax_) {
      ++Ntheta_;
      thetamax_ = theta_[i];
    }
  }



  /* // DEBUGG
  for (int ir=0; ir<MPI::COMM_WORLD.Get_size(); ++ir)
  {
    if (rank==ir)
    {
      for (size_t i=0; i<a_.size(); ++i)
      {
	std::cout << std::setw(15) << rank
		  << std::setw(15) << i
		  << std::setw(15) << a_[i]
		  << std::setw(15) << theta_[i]
		  << std::setw(15) << ne_[i]
		  << std::endl;
      }
      std::cout << std::endl;
    }
    MPI::COMM_WORLD.Barrier();
  }
  */
}
void SgrA_PolintDiskModelParameters2010::reset(double a, double THETA)
{
  _a = a;
  _THETA = THETA;
}

void SgrA_PolintDiskModelParameters2010::set_orders(int aorder, int thetaorder)
{
  _aorder = aorder;
  _thetaorder = thetaorder;
}

double SgrA_PolintDiskModelParameters2010::ne_norm() const
{
  return (  std::pow(10.0, interpolate2D(ne_,_aorder,_thetaorder,_a,_THETA) )  );
}
double SgrA_PolintDiskModelParameters2010::Te_norm() const
{
  return (  std::pow(10.0, interpolate2D(Te_,_aorder,_thetaorder,_a,_THETA) )  );
}
double SgrA_PolintDiskModelParameters2010::nnth_norm() const
{
  return (  std::pow(10.0, interpolate2D(nnth_,_aorder,_thetaorder,_a,_THETA) )  );
}


double SgrA_PolintDiskModelParameters2010::interpolate2D(const std::vector<double>& y, int na, int nth, double a, double theta) const
{
  // 1st get interpolation limits
  //  (A) Get stencil size to fill
  if (nth>Ntheta_) // Can only use the maximum number of points
    nth = Ntheta_;
  if (na>Na_)
    na = Na_;
  //  (B) Assuming that a_ & theta_ are in ascending order, get index of first value less than one of interest
  int ith=0, ia=0;
  for (ith=0; ith<Ntheta_ && theta>theta_[ith]; ++ith) {};
  for (ia=0; ia<Na_ && a>a_[ia*Ntheta_]; ++ia) {};
  //  (C) Try to center stencil on this value
  int ith0 = std::min(std::max(ith-nth/2,0),Ntheta_-nth);
  int ia0 = std::min(std::max(ia-na/2,0),Na_-na);

  // Allocate work arrays
  double *xth = new double[nth+1];
  double *yth = new double[nth+1];
  double *xa = new double[na+1];
  double *ya = new double[na+1];

  // Do interpolations
  int index_offset;
  double dy;
  for (ia=0; ia<na; ++ia) {
    index_offset = (ia0+ia)*Ntheta_;
    // (A) For each a in stencil interpolate on theta
    for (ith=0; ith<nth; ++ith) {
      xth[ith+1] = theta_[index_offset + ith0 + ith];
      yth[ith+1] = y[index_offset + ith0 + ith];
    }
    polint(xth,yth,nth,theta,ya[ia+1],dy);
    xa[ia+1] = a_[index_offset];
  }
  double val;
  polint(xa,ya,na,a,val,dy);

  return val;
}

// Numerical Recipes: Interpolates x into the arrays xa,ya to get y with
//   error estimate dy after doing nth-order polynomial interpolation.
// NOTE THAT THIS USES UNIT-OFFSET ARRAYS (YUCK!!!)
void SgrA_PolintDiskModelParameters2010::polint(double xa[], double ya[], int n, double x, double &y, double &dy) const
{
  int i,m,ns=1;
  double den,dif,dift,ho,hp,w;
  double *c=new double[n+1];
  double *d=new double[n+1];

  dif=fabs(x-xa[1]);
  for (i=1;i<=n;i++) {
    if ( (dift=fabs(x-xa[i])) < dif) {
      ns=i;
      dif=dift;
    }
    c[i]=ya[i];
    d[i]=ya[i];
  }
  y=ya[ns--];
  for (m=1;m<n;m++) {
    for (i=1;i<=n-m;i++) {
      ho=xa[i]-x;
      hp=xa[i+m]-x;
      w=c[i+1]-d[i];
      if ( (den=ho-hp) == 0.0)
	std::cerr << "Error in routine polint\n";
      den=w/den;
      d[i]=hp*den;
      c[i]=ho*den;
    }
    y += (dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
  }
  delete[] c;
  delete[] d;
}

};
