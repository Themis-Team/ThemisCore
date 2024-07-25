#include "SgrA_disk_model_parameters_ht_2011.h"

namespace VRT2 {
SgrA_PolintDiskModelParametersHT2011::SgrA_PolintDiskModelParametersHT2011(std::string fname, int aorder, int thetaorder, int eorder, double aval, double THETAval, double eval)
  : _a(aval), _THETA(THETAval), _e(eval), _aorder(aorder), _thetaorder(thetaorder), _eorder(eorder)
{
#ifdef MPI_MAP
  int rank = MPI::COMM_WORLD.Get_rank();
#else
  int rank = 0;
#endif

  // Read data from file into tables (proc 0 only!)
  std::vector<int> ia, it, ie;
  std::vector<double> a, t, e, ne, Te, nnth;
  
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
      int itmp;
      double tmp;
      in >> itmp;
      if (!in.eof())
      {
	ia.push_back(itmp);
	in >> itmp;
	it.push_back(itmp);
	in >> itmp;
	ie.push_back(itmp);
	in >> tmp;
	a.push_back(tmp);
	in >> tmp;
	t.push_back(tmp);
	in >> tmp;
	e.push_back(tmp);
	in >> tmp;
	ne.push_back(tmp);
	in >> tmp;
	Te.push_back(tmp);
	in >> tmp;
	nnth.push_back(tmp);
	in.ignore(4096,'\n');
      }
    } while (!in.eof());

  }
#ifdef MPI_MAP
  // Share if necessary
  int tablesize;
  if (rank==0)
    tablesize=ia.size();
  MPI::COMM_WORLD.Bcast(&tablesize,1,MPI::INT,0);
  int *ibuff = new int[3*tablesize];
  double *buff = new double[6*tablesize];
  if (rank==0)
    for (int i=0; i<tablesize; i++)
    {
      ibuff[3*i+0] = ia[i];
      ibuff[3*i+1] = it[i];
      ibuff[3*i+2] = ie[i];
      buff[6*i+0] = a[i];
      buff[6*i+1] = t[i];
      buff[6*i+2] = e[i];
      buff[6*i+3] = ne[i];
      buff[6*i+4] = Te[i];
      buff[6*i+5] = nnth[i];
    }
  MPI::COMM_WORLD.Bcast(&ibuff[0],3*tablesize,MPI::INT,0);
  MPI::COMM_WORLD.Bcast(&buff[0],6*tablesize,MPI::DOUBLE,0);
  if (rank!=0)
  {
    ia.resize(tablesize);
    it.resize(tablesize);
    ie.resize(tablesize);
    a.resize(tablesize);
    t.resize(tablesize);
    e.resize(tablesize);
    ne.resize(tablesize);
    Te.resize(tablesize);
    nnth.resize(tablesize);
    for (int i=0; i<tablesize; i++)
    {
      ia[i] = ibuff[3*i+0];
      it[i] = ibuff[3*i+1];
      ie[i] = ibuff[3*i+2];
      a[i] = buff[6*i+0];
      t[i] = buff[6*i+1];
      e[i] = buff[6*i+2];
      ne[i] = buff[6*i+3];
      Te[i] = buff[6*i+4];
      nnth[i] = buff[6*i+5];
    }
  }
  delete[] ibuff;
  delete[] buff;
#endif


  // Sort out table details



  // Get all of the e's
  emin_ = e[0];
  int iemin = ie[0];
  for (size_t i=1; i<ie.size(); ++i)
  {
    emin_ = std::min(e[i],emin_);
    iemin = std::min(ie[i],iemin);
  }
  Ne_=1;
  int iemax = iemin;
  for (size_t i=0; i<ie.size(); ++i)
  {
    if (ie[i]>iemax)
    {
      ++Ne_;
      emax_ = e[i];
      iemax = ie[i];
    }
  }
  e_ = new double[Ne_];
  for (size_t i=0; i<ie.size(); ++i)
    e_[ie[i]-iemin] = e[i];
  
  
  // Get the e-specific a-theta subgrids
  amin_ = new double[Ne_];
  amax_ = new double[Ne_];
  tmin_ = new double[Ne_];
  tmax_ = new double[Ne_];
  Na_ = new int[Ne_];
  Nt_ = new int[Ne_];
  
  a_ = new double*[Ne_];
  t_ = new double*[Ne_];


  int *iamin = new int[Ne_];
  int *itmin = new int[Ne_];
  for (int j=0; j<Ne_; ++j)
  {
    int imin, imax;
    bool setmin=false;
    // Get all of the a's
    for (size_t i=0; i<ia.size(); ++i)
    {
      if (ie[i]==iemin+j)
      {
	if (!setmin) {
	  amin_[j] = a[i];
	  imin = ia[i];
	  setmin=true;
	}
	amin_[j] = std::min(a[i],amin_[j]);
	imin = std::min(ia[i],imin);
      }
    }
    Na_[j]=1;
    iamin[j] = imax = imin;
    for (size_t i=0; i<ia.size(); ++i)
    {
      if (ie[i]==iemin+j)
      {
	if (ia[i]>imax)
	{
	  ++Na_[j];
	  amax_[j] = a[i];
	  imax = ia[i];
	}
      }
    }
    a_[j] = new double[Na_[j]];
    for (size_t i=0; i<ia.size(); ++i)
      if (ie[i]==iemin+j)
	a_[j][ia[i]-imin] = a[i];
    
    // Get all of the t's
    setmin=false;
    for (size_t i=0; i<it.size(); ++i)
    {
      if (ie[i]==iemin+j)
      {
	if (!setmin) {
	  tmin_[j] = t[i];
	  imin = it[i];
	  setmin=true;
	}
	tmin_[j] = std::min(t[i],tmin_[j]);
	imin = std::min(it[i],imin);
      }
    }
    Nt_[j]=1;
    itmin[j] = imax = imin;
    for (size_t i=0; i<it.size(); ++i)
    {
      if (ie[i]==iemin+j)
      {
	if (it[i]>imax)
	{
	  ++Nt_[j];
	  tmax_[j] = t[i];
	  imax = it[i];
	}
      }
    }
    t_[j] = new double[Nt_[j]];
    for (size_t i=0; i<it.size(); ++i)
      if (ie[i]==iemin+j)
	t_[j][it[i]-imin] = t[i];
    
  }

  ne_ = new double**[Ne_];
  Te_ = new double**[Ne_];
  nnth_ = new double**[Ne_];
  for (int i=0; i<Ne_; ++i)
  {
    ne_[i] = new double*[Na_[i]];
    Te_[i] = new double*[Na_[i]];
    nnth_[i] = new double*[Na_[i]];
    for (int j=0; j<Na_[i];  ++j)
    {
      ne_[i][j] = new double[Nt_[i]];
      Te_[i][j] = new double[Nt_[i]];
      nnth_[i][j] = new double[Nt_[i]];
    }
  }
    
  for (size_t i=0; i<ia.size(); ++i)
  {
    ne_[ie[i]-iemin][ia[i]-iamin[ie[i]-iemin]][it[i]-itmin[ie[i]-iemin]] = ne[i];
    Te_[ie[i]-iemin][ia[i]-iamin[ie[i]-iemin]][it[i]-itmin[ie[i]-iemin]] = Te[i];
    nnth_[ie[i]-iemin][ia[i]-iamin[ie[i]-iemin]][it[i]-itmin[ie[i]-iemin]] = nnth[i];
  }
  delete[] iamin;
  delete[] itmin;

  /*
  for (int i=0; i<Ne_; ++i)
    for (int j=0; j<Na_[i]; ++j)
      for (int k=0; k<Nt_[i]; ++k)
	std::cout << std::setw(15) << a_[i][j]
		  << std::setw(15) << t_[i][k]
		  << std::setw(15) << e_[i]
		  << std::setw(15) << ne_[i][j][k]
		  << std::setw(15) << Te_[i][j][k]
		  << std::setw(15) << nnth_[i][j][k]
		  << std::endl;
  */

}

SgrA_PolintDiskModelParametersHT2011::~SgrA_PolintDiskModelParametersHT2011()
{

  for (int i=0; i<Ne_; ++i)
  {
    for (int j=0; j<Na_[i];  ++j)
    {
      delete[] ne_[i][j];
      delete[] Te_[i][j];
      delete[] nnth_[i][j];
    }
    delete[] ne_[i];
    delete[] Te_[i];
    delete[] nnth_[i];
    delete[] a_[i];
    delete[] t_[i];
  }
  delete[] ne_;
  delete[] Te_;
  delete[] nnth_;
  delete[] e_;
  delete[] amin_;
  delete[] amax_;
  delete[] tmin_;
  delete[] tmax_;
  delete[] Na_;
  delete[] Nt_;
}

int SgrA_PolintDiskModelParametersHT2011::index(int ie, int it, int ia) const
{
  return ( ia + Na_[ie]*(it + Nt_[ie]*ie) );
}
void SgrA_PolintDiskModelParametersHT2011::reset(double a, double THETA, double e)
{
  _a = a;
  _THETA = THETA;
  _e = e;
}

void SgrA_PolintDiskModelParametersHT2011::set_orders(int aorder, int thetaorder, int ie)
{
  _aorder = aorder;
  _thetaorder = thetaorder;
  _eorder = ie;
}

double SgrA_PolintDiskModelParametersHT2011::ne_norm() const
{
  return (  std::pow(10.0, interpolate3D(ne_,_aorder,_thetaorder,_eorder,_a,_THETA,_e))  ); 
  //return 0;
  //return (  std::pow(10.0, interpolate2D(ne_,_aorder,_thetaorder,_a,_THETA) )  );
}
double SgrA_PolintDiskModelParametersHT2011::Te_norm() const
{
  return (  std::pow(10.0, interpolate3D(Te_,_aorder,_thetaorder,_eorder,_a,_THETA,_e))  ); 
  //return 0;
  //return (  std::pow(10.0, interpolate2D(Te_,_aorder,_thetaorder,_a,_THETA) )  );
}
double SgrA_PolintDiskModelParametersHT2011::nnth_norm() const
{
  return (  std::pow(10.0, interpolate3D(nnth_,_aorder,_thetaorder,_eorder,_a,_THETA,_e))  ); 
  //return 0;
  //return (  std::pow(10.0, interpolate2D(nnth_,_aorder,_thetaorder,_a,_THETA) )  );
}

double SgrA_PolintDiskModelParametersHT2011::interpolate3D(double ***fg, int npa, int npt, int npe, double a, double t, double e) const
{
  // Need a somewhat intelligent stencil sizer to account for
  // missing parameter space in the e-direction
  int npemax = Ne_;
  for (int i=0; i<Ne_; ++i)
    if (a>amax_[i])
      npemax--;
  if (npe>npemax)
    npe = npemax;

  // Locate the relevant ie
  int ie=0;
  if (e_[Ne_-1]-e_[0]>0)
  {
    for (ie=0; ie<Ne_ && e>e_[ie]; ++ie) {};
  }
  else
  {
    for (ie=Ne_-1; ie>=0 && e>=e_[ie]; --ie) {};
  }
  if (a>amax_[ie]) // I.e., we're in the weeds
    npe++; // Let's extrapolate in all directions, and we know we can add since we must have subtracted at least one!



  // Try to center stencil on this value, USES FACT THAT THE a-RANGE FOR SMALL e IS SMALLER
  int ie0 = std::min(std::max(ie-npe/2,Ne_-npemax),Ne_-npe);



  // work with unit offset
  double *xe = new double[npe+1];
  double *ye = new double[npe+1];

  // Do interpolations
  double dy;
  for (ie=0; ie<npe; ++ie)
  {
    ye[ie+1] = interpolate2D(fg[ie+ie0],a_[ie+ie0],t_[ie+ie0],Na_[ie+ie0],Nt_[ie+ie0],npa,npt,a,t);
    xe[ie+1] = e_[ie+ie0];
  }
  double val;
  polint(xe,ye,npe,e,val,dy);

  /*
  double val2 = interpolate2D(fg[4],a_[4],t_[4],Na_[4],Nt_[4],npa,npt,a,t);
  std::cerr << "ie0 = " << ie0
	    << "  Ne_ = " << Ne_
	    << "  npe = " << npe
	    << "  npemax = " << npemax
	    << "  ie-npe/2 = " << ie-npe/2
	    << "  Ne_-npemax = " << Ne_-npemax
	    << "  Ne_-npe = " << Ne_-npe;

  for (ie=1; ie<=npe; ++ie)
    std::cerr << std::setw(15) << xe[ie];
  for (ie=1; ie<=npe; ++ie)
    std::cerr << std::setw(15) << ye[ie];
  std::cerr << '|'
	    << std::setw(15) << e
	    << std::setw(15) << val
	    << std::setw(15) << val2
	    << '\n';
  */


  delete[] xe;
  delete[] ye;



  return val;
}

double SgrA_PolintDiskModelParametersHT2011::interpolate2D(double **fg, const double *xg, const double *yg, int Nxg, int Nyg, int npx, int npy, double x, double y) const
{
  // (A) Get stencil size to fill, can only use the maximum number of points
  if (npx>Nxg)
    npx = Nxg;
  if (npy>Nyg)
    npy = Nyg;

  // (B) Get the index of the first value bracketing the one of interest.
  int ix=0, iy=0;
  if (xg[Nxg-1]-xg[0]>0)
  {
    for (ix=0; ix<Nxg && x>xg[ix]; ++ix) {};
  }
  else
  {
    for (ix=Nxg-1; ix>=0 && x>=xg[ix]; --ix) {};
  }
  
  if (yg[Nyg-1]-yg[0]>0)
  {
    for (iy=0; iy<Nyg && y>yg[iy]; ++iy) {};
  }
  else
  {
    for (iy=Nyg-1; iy>=0 && y>=yg[iy]; --iy) {};
  }

  // Try to center stencil on this value
  int ix0 = std::min(std::max(ix-npx/2,0),Nxg-npx);
  int iy0 = std::min(std::max(iy-npy/2,0),Nyg-npy);

  // Allocate work arrays (with unit offset!)
  double *xx = new double[npx+1];
  double *yx = new double[npx+1];
  double *xy = new double[npy+1];
  double *yy = new double[npy+1];

  // Do interpolations
  double dy;
  for (ix=0; ix<npx; ++ix)
  {
    for (iy=0; iy<npy; ++iy)
    {
      xy[iy+1] = yg[iy+iy0];
      yy[iy+1] = fg[ix+ix0][iy+iy0];
    }
    polint(xy,yy,npy,y,yx[ix+1],dy);
    xx[ix+1] = xg[ix+ix0];
  }
  double val;
  polint(xx,yx,npx,x,val,dy);

  delete[] xx;
  delete[] yx;
  delete[] xy;
  delete[] yy;


  return val;
}


// I.e. Interpolate1D!
// Numerical Recipes: Interpolates x into the arrays xa,ya to get y with
//   error estimate dy after doing nth-order polynomial interpolation.
// NOTE THAT THIS USES UNIT-OFFSET ARRAYS (YUCK!!!)
void SgrA_PolintDiskModelParametersHT2011::polint(double xa[], double ya[], int n, double x, double &y, double &dy) const
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
