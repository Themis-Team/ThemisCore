#include "M87_jet_model_parameters_2012.h"

namespace VRT2 {

M87_PolintJetModelParameters2012::M87_PolintJetModelParameters2012(std::string fname, int corder, int aorder, int rorder, double cval, double aval, double rval)
: _c(cval), _a(aval), _r(rval), _corder(corder), _aorder(aorder), _rorder(rorder)
{
#ifdef MPI_MAP
  int rank = MPI::COMM_WORLD.Get_rank();
#else
  int rank = 0;
#endif

  // Read data from file into tables (proc 0 only!)
  std::vector<int> ic, ia, ir;
  std::vector<double> c, a, r, nj, bj, aj;
  
  if (rank==0)
  {


    std::ifstream in(fname.c_str());
    if (!in.is_open())
    {
      std::cerr << "Couldn't open " << fname << std::endl;
      std::exit(1);
    }

    // HAVE TO DECIDE ON HEADERS!
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
	ir.push_back(itmp);
	in >> itmp;
	ic.push_back(itmp);
	in >> tmp;
	a.push_back(tmp);
	in >> tmp;
	r.push_back(tmp);
	in >> tmp;
	c.push_back(tmp);
	in >> tmp;
	nj.push_back(tmp);
	in >> tmp;
	bj.push_back(tmp);
	in >> tmp;
	aj.push_back(tmp);
	in.ignore(4096,'\n');
      }
    } while (!in.eof());

  }
#ifdef MPI_MAP
  // Share if necessary
  int tablesize;
  if (rank==0)
    tablesize=ic.size();
  MPI::COMM_WORLD.Bcast(&tablesize,1,MPI::INT,0);
  int *ibuff = new int[3*tablesize];
  double *buff = new double[6*tablesize];
  if (rank==0)
    for (int i=0; i<tablesize; i++)
    {
      ibuff[3*i+0] = ic[i];
      ibuff[3*i+1] = ia[i];
      ibuff[3*i+2] = ir[i];
      buff[6*i+0] = c[i];
      buff[6*i+1] = a[i];
      buff[6*i+2] = r[i];
      buff[6*i+3] = nj[i];
      buff[6*i+4] = bj[i];
      buff[6*i+5] = aj[i];
    }
  MPI::COMM_WORLD.Bcast(&ibuff[0],3*tablesize,MPI::INT,0);
  MPI::COMM_WORLD.Bcast(&buff[0],6*tablesize,MPI::DOUBLE,0);
  if (rank!=0)
  {
    ic.resize(tablesize);
    ia.resize(tablesize);
    ir.resize(tablesize);
    c.resize(tablesize);
    a.resize(tablesize);
    r.resize(tablesize);
    nj.resize(tablesize);
    bj.resize(tablesize);
    aj.resize(tablesize);
    for (int i=0; i<tablesize; i++)
    {
      ic[i] = ibuff[3*i+0];
      ia[i] = ibuff[3*i+1];
      ir[i] = ibuff[3*i+2];
      c[i] = buff[6*i+0];
      a[i] = buff[6*i+1];
      r[i] = buff[6*i+2];
      nj[i] = buff[6*i+3];
      bj[i] = buff[6*i+4];
      aj[i] = buff[6*i+5];
    }
  }
  delete[] ibuff;
  delete[] buff;
#endif


  // Sort out table details


  // Get the range in c
  int icmin, icmax;
  cmin_ = cmax_ = c[0];
  icmin = icmax = ic[0];
  for (size_t i=1; i<ic.size(); ++i)
  {
    cmin_ = std::min(c[i],cmin_);
    icmin = std::min(ic[i],icmin);
    cmax_ = std::max(c[i],cmax_);
    icmax = std::max(ic[i],icmax);
  }
  Nc_ = icmax-icmin+1;
  c_ = new double[Nc_];
  for (size_t i=0; i<ic.size(); ++i)
    c_[ic[i]-icmin] = c[i];

  // Get the range in a
  int iamin, iamax;
  amin_ = amax_ = a[0];
  iamin = iamax = ia[0];
  for (size_t i=1; i<ia.size(); ++i)
  {
    amin_ = std::min(a[i],amin_);
    iamin = std::min(ia[i],iamin);
    amax_ = std::max(a[i],amax_);
    iamax = std::max(ia[i],iamax);
  }
  Na_ = iamax-iamin+1;
  a_ = new double[Na_];
  for (size_t i=0; i<ia.size(); ++i)
    //a_[ia[i]-iamin] = a[i];
    a_[ia[i]-iamin] = std::log(1.0-a[i]);
 
  // Get the a-specific rl subgrid
  rmin_ = new double[Na_];
  rmax_ = new double[Na_];
  Nr_ = new int[Na_];
  r_ = new double*[Na_];
  int *irmin = new int[Na_];
  for (int j=0; j<Na_; ++j)
  {
    int imin, imax;

    // Get rl's
    rmin_[j] = rmax_[j] = r[0];
    imin = imax = ir[0];
    for (size_t i=1; i<ir.size(); ++i)
    {
      if (ia[i]==iamin+j)
      {
	rmin_[j] = std::min(r[i],rmin_[j]);
	imin = std::min(ir[i],imin);
	rmax_[j] = std::max(r[i],rmax_[j]);
	imax = std::max(ir[i],imax);
      }
    }
    Nr_[j] = imax-imin+1;
    r_[j] = new double[Nr_[j]];
    for (size_t i=0; i<ir.size(); ++i)
      r_[j][ir[i]-imin] = r[i];
    irmin[j] = imin;
  }

  nj_ = new double**[Na_];
  bj_ = new double**[Na_];
  aj_ = new double**[Na_];
  for (int i=0; i<Na_; ++i)
  {
    nj_[i] = new double*[Nc_];
    bj_[i] = new double*[Nc_];
    aj_[i] = new double*[Nc_];
    for (int j=0; j<Nc_;  ++j)
    {
      nj_[i][j] = new double[Nr_[i]];
      bj_[i][j] = new double[Nr_[i]];
      aj_[i][j] = new double[Nr_[i]];
    }
  }
    
  for (size_t i=0; i<ic.size(); ++i)
  {
    nj_[ia[i]-iamin][ic[i]-icmin][ir[i]-irmin[ia[i]-iamin]] = nj[i];
    bj_[ia[i]-iamin][ic[i]-icmin][ir[i]-irmin[ia[i]-iamin]] = bj[i];
    aj_[ia[i]-iamin][ic[i]-icmin][ir[i]-irmin[ia[i]-iamin]] = aj[i];
  }
  delete[] irmin;

  /*
  // Check the read in
  std::cerr << std::setw(15) << iamin
	    << std::setw(15) << icmin
	    << std::setw(15) << irmin[0]
	    << std::setw(15) << irmin[1]
	    << std::endl;
  std::cerr << std::setw(15) << Na_
	    << std::setw(15) << Nc_
	    << std::setw(15) << Nr_[0]
	    << std::setw(15) << Nr_[1]
	    << std::endl;
  for (size_t j=0; j<Nc_; ++j)
    for (size_t k=0; k<Na_; ++k)
      for (size_t l=0; l<Nr_[k]; ++l)
	std::cout << std::setw(15) << k
		  << std::setw(15) << l
		  << std::setw(15) << j
		  << std::setw(15) << a_[k]
		  << std::setw(15) << r_[k][l]
		  << std::setw(15) << c_[j]
		  << std::setw(15) << nj_[k][j][l]
		  << std::setw(15) << bj_[k][j][l]
		  << std::setw(15) << aj_[k][j][l]
		  << std::endl;
  std::cout << "\n" << std::endl;
  */
}

M87_PolintJetModelParameters2012::~M87_PolintJetModelParameters2012()
{

  for (int i=0; i<Na_; ++i)
  {
    for (int j=0; j<Nc_;  ++j)
    {
      delete[] nj_[i][j];
      delete[] bj_[i][j];
      delete[] aj_[i][j];
    }
    delete[] nj_[i];
    delete[] bj_[i];
    delete[] aj_[i];
    delete[] r_[i];
  }
  delete[] nj_;
  delete[] bj_;
  delete[] aj_;
  delete[] c_;
  delete[] r_;
  delete[] a_;
  delete[] rmin_;
  delete[] rmax_;
  delete[] Nr_;
}

void M87_PolintJetModelParameters2012::reset(double c, double a, double r)
{
  _a = a;
  _c = c;
  _r = r;
}

void M87_PolintJetModelParameters2012::set_orders(int corder, int aorder, int rorder)
{
  _corder = corder;
  _aorder = aorder;
  _rorder = rorder;
}

double M87_PolintJetModelParameters2012::nj_norm() const
{
  return (  std::pow(10.0, interpolate3D(nj_,_corder,_aorder,_rorder,_c,_a,_r))  ); 
}
double M87_PolintJetModelParameters2012::bj_norm() const
{
  return (  std::pow(10.0, interpolate3D(bj_,_corder,_aorder,_rorder,_c,_a,_r))  ); 
}
double M87_PolintJetModelParameters2012::nj_index() const
{
  return (  std::max(0.5, interpolate3D(aj_,_corder,_aorder,_rorder,_c,_a,_r))  ); 
}

double M87_PolintJetModelParameters2012::disk_inner_edge_radius() const
{
  Kerr g(1.0,_a);
  return g.rISCO();
}

double M87_PolintJetModelParameters2012::interpolate3D(double ***fg, int npc, int npa, int npr, double c, double a, double r) const
{
  //double aorig = a;
  a = std::log(1.0-a);

  // Interpolation is done in two steps:
  //   1. Interpolating on a square R-c grid at a number of a to get f(a)
  //   2. Interpolating on the f(a) to get the desired value
  //
  // We first must define the smallest a for which the table includes the relevant points.
  // This assumes that the table is simply connected.
  int iamin=0, iamax=Na_-1;
  for (int i=0; i<Na_; ++i)
  {
    if (r<rmin_[i]) // If r is less than the minimum value, don't include this a-grid
      iamin++;
    if (r>rmax_[i]) // If r is more than the maximum value, don't include this a-grid
      iamax--;
  }

  int ia;
  if (a_[Na_-1]-a_[0]>0) // If increasing
  {
    for (ia=iamin; ia<iamax && a_[ia]<a; ++ia) {};
    //ia--;
  }
  else
  {
    for (ia=iamax; ia>=iamin && a_[ia]<a; --ia) {};
    ia++;
  }
  if (npa>iamax-iamin)
    npa = iamax-iamin;

  //int iafound = ia;

  int ia0;
  if (a_[Na_-1]-a_[0]>0) // If increasing
    ia0 = std::min( std::max(ia-npa/2+npa%2,iamin), iamax+1-npa );
  else
    ia0 = std::min( std::max(ia-npa/2+npa%2,iamin), iamax+1-npa );


  // work with unit offset
  double *xa = new double[npa+1];
  double *ya = new double[npa+1];

  // Do interpolations to fill f(a)
  double dy;
  for (ia=0; ia<npa; ++ia)
  {
    ya[ia+1] = interpolate2D(fg[ia+ia0],c_,r_[ia+ia0],Nc_,Nr_[ia+ia0],npc,npr,c,r);
    xa[ia+1] = a_[ia+ia0];
  }
  double val;
  polint(xa,ya,npa,a,val,dy);

  /*
  if (aorig>0.8)
  {
    std::cout << std::setw(15) << aorig
	      << std::setw(15) << a
	      << std::setw(15) << iamin
	      << std::setw(15) << iamax
	      << std::setw(15) << iafound
	      << std::setw(15) << iafound-npa/2
	      << std::setw(15) << iafound-npa/2+npa%2
	      << std::setw(15) << ia0
	      << std::setw(15) << npa;
    std::cout << "  |  ";
    for (ia=0; ia<npa; ++ia)
      std::cout << std::setw(15) << xa[ia+1];
    std::cout << "  |  ";
    for (ia=0; ia<npa; ++ia)
      std::cout << std::setw(15) << ya[ia+1];
    std::cout << "  |  ";
    std::cout << std::setw(15) << val
	      << std::endl;
  }
  */

  delete[] xa;
  delete[] ya;



  return val;
}




double M87_PolintJetModelParameters2012::interpolate2D(double **fg, const double *xg, const double *yg, int Nxg, int Nyg, int npx, int npy, double x, double y) const
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
void M87_PolintJetModelParameters2012::polint(double xa[], double ya[], int n, double x, double &y, double &dy) const
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
