// Include Statements
#include "metric.h"

namespace VRT2 {

// Constructor
Metric::Metric(int Ng_val, int NDg_val, int NG_val)
  : Ng(Ng_val), NDg(NDg_val), NG(NG_val), _x(0.0,4)
{ 
  defined_list.reset();
}

void Metric::initialize()
{
}

void Metric::mk_hash_table()
{
  for (int i=0; i<4; ++i)
    for (int j=0; j<4; ++j) {
      _gi[i][j] = -1;
      for (int k=0; k<Ng; ++k)
	if ( (gi[k]==i && gj[k]==j) || (gi[k]==j && gj[k]==i) )
	  _gi[i][j] = k;
      for (int k=0; k<4; ++k) {
	_Dgi[i][j][k] = -1;
	for (int l=0; l<NDg; ++l)
	  if ( (Dgi[l]==i && Dgj[l]==j && Dgk[l]==k)
	       || (Dgi[l]==j && Dgj[l]==i && Dgk[l]==k) )
	    _Dgi[i][j][k] = l;
	_Gi[i][j][k] = -1;
	for (int l=0; l<NG; ++l)
	  if ( (Gi[l]==i && Gj[l]==j && Gk[l]==k)
	       || (Gi[l]==i && Gj[l]==k && Gk[l]==j) )
	    _Gi[i][j][k] = l;
      
      }
    }
}

/*** Elements by i,j etc. (slow) ***/
// g_ij
double Metric::g(int i, int j)
{
  if (_gi[i][j]>=0) {
    return ( g(_gi[i][j]) );
  }
  else
    return 0;
}

// g^ij
double Metric::ginv(int i, int j)
{
  if (_gi[i][j]>=0)
    return ( ginv(_gi[i][j]) );
  else
    return 0;
}

// Dg_ij,k
double Metric::Dg(int i, int j, int k)
{
  if (_Dgi[i][j][k]>=0)
    return ( Dg(_Dgi[i][j][k]) );
  else
    return 0;
}

// Dg^ij,k
double Metric::Dginv(int i, int j, int k)
{
  if (_Dgi[i][j][k]>=0)
    return ( Dginv(_Dgi[i][j][k]) );
  else
    return 0;
}

// Gamma^i_jk
double Metric::Gamma(int i, int j, int k)
{
  if (_Gi[i][j][k]>=0)
    return ( Gamma(_Gi[i][j][k]) );
  else
    return 0;
}

// Position dependent funcs used by g and ginv
void Metric::get_fcns()
{
}

void Metric::reset(const double x[])
{
  defined_list.reset();
  for(int i1=0;i1<4;++i1)
    _x[i1] = x[i1];
  get_fcns();
}

void Metric::reset(double x0, double x1, double x2, double x3)
{
  defined_list.reset();
  _x[0] = x0;
  _x[1] = x1;
  _x[2] = x2;
  _x[3] = x3;
  get_fcns();
}

void Metric::reset(const std::valarray<double> &x)  
{
  defined_list.reset();
  _x = x;
  get_fcns();
}

void Metric::reset(const std::vector<double>& x)  
{
  defined_list.reset();
  for(int i1=0;i1<4;++i1)
    _x[i1] = x[i1];
  get_fcns();
}

// Horizon
double Metric::horizon() const
{
  return ( 0.0 );				
}

// Optimization check
void Metric::check_compiler_optimization()
{
  bool bad_metric = false;

#if defined(BOYER_LINDQUIST_LIKE_METRIC)

  if (Ng!=6)
    bad_metric = bad_metric || true;
  for (int k=0; k<Ng-2; ++k)
    if ( !(gi[k]==k && gj[k]==k) )
      bad_metric = bad_metric || true;
  for (int k=Ng-2; k<Ng; ++k)
    if (!( gi[k]==0 && gj[k]==3 ) && !( gi[k]==3 && gj[k]==0 ))
      bad_metric = bad_metric || true;
  if (bad_metric) {
    std::cerr << "VRT2 ERROR: libvrt2.a has been compiled with DIAGONAL_METRIC set, but non-diagonal metric detected!  (Note that additional structure restrictions apply when DIAGONAL_METRIC is specified.  See operator*(FourVector...) in fourvector.h for more details.)  Exiting now.\n\n";
    std::exit(1);
  }
  return;

#elif defined(DIAGONAL_METRIC)

  if (Ng!=4)
    bad_metric = bad_metric || true;
  for (int k=0; k<Ng; ++k)
    if ( !(gi[k]==k && gj[k]==k) )
      bad_metric = bad_metric || true;
  if (bad_metric) {
    std::cerr << "VRT2 ERROR: libvrt2.a has been compiled with DIAGONAL_METRIC set, but non-diagonal metric detected!  (Note that additional structure restrictions apply when DIAGONAL_METRIC is specified.  See operator*(FourVector...) in fourvector.h for more details.)  Exiting now.\n\n";
    std::exit(1);
  }
  return;

#endif


}


void Metric::derivatives_check(double x0[], double h)
{
  std::cout << std::endl << std::endl
	    << "*** Metric Derivative Check ***********************"
	    << std::endl << std::endl;


  // Reinitialize
  reset(x0);

  double x[4];
  double g0, dgdx_fd;
  int ig=0;
  
  for (int i=0; i<4; ++i)
    x[i] = x0[i];

  // g_ab,c
  for (int i=0; i<NDg; ++i) {
    // Output analytic derivatives
    reset(x0);
    std::cout << "    dg[" << Dgi[i] << ',' << Dgj[i] << "]dx["
	      << Dgk[i] << "]_an = " << std::setw(15) << Dg(i)
	      << std::endl;
 
    // Forward difference
    // Get g0
    reset(x0);
    for (int j=0; j<Ng; ++j)
      if (gi[j]==Dgi[i] && gj[j]==Dgj[i])
	ig = j;
    g0 = g(ig);
    // Make step
    x[Dgk[i]] += h;
    // Reinitialize at new step
    reset(x);
    // Calc. foward diff.
    dgdx_fd = (g(ig)-g0)/h;
    // Reset step
    x[Dgk[i]] = x0[Dgk[i]];
    // Output foward diff.
    std::cout << "    dg[" << Dgi[i] << ',' << Dgj[i] << "]dx["
	      << Dgk[i] << "]_fd = " << std::setw(15) << dgdx_fd
	      << std::endl << std::endl;
  }
  std::cout << std::endl << std::endl;


  // g^ab_,c
  for (int i=0; i<NDg; ++i) {
    // Output analytic derivatives
    reset(x0);
    std::cout << "    dg[" << Dgi[i] << ',' << Dgj[i] << "]dx["
	      << Dgk[i] << "]_an = " << std::setw(15) << Dginv(i)
	      << std::endl;
 
    // Foward difference
    // Get g0
    reset(x0);
    for (int j=0; j<Ng; ++j)
      if (gi[j]==Dgi[i] && gj[j]==Dgj[i])
	ig = j;
    g0 = ginv(ig);
    // Make step
    x[Dgk[i]] += h;
    // Reinitialize at new step
    reset(x);
    // Calc. foward diff.
    dgdx_fd = (ginv(ig)-g0)/h;
    // Reset step
    x[Dgk[i]] = x0[Dgk[i]];
    // Output foward diff.
    std::cout << "    dg[" << Dgi[i] << ',' << Dgj[i] << "]dx["
	      << Dgk[i] << "]_fd = " << std::setw(15) << dgdx_fd
	      << std::endl << std::endl;
  }
  std::cout << std::endl << std::endl
	    << "***************************************************"
	    << std::endl << std::endl;
}

std::ostream& operator<<(std::ostream& os, Metric& metric)
{
  os.setf(std::ios::scientific);
  os << std::setprecision(3);

  // Output covariant metric
  for (int i1=0;i1<metric.Ng;i1++){
    if (i1)
      os << "  "; // Spacer
    os << "g_" << metric.gi[i1] << metric.gj[i1] << " = " 
       << std::setw(10) << metric.g(i1);
  }

  os << std::endl;

  // Output contravariant metric
  for (int i1=0;i1<metric.Ng;i1++){
    if (i1)
      os << "  "; // Spacer
    os << "g^" << metric.gi[i1] << metric.gj[i1] << " = " 
       << std::setw(10) << metric.ginv(i1);
  }

  os << std::endl;

  // Output g_ij,k
  for (int i1=0;i1<metric.NDg;i1++){
    if (i1)
      os << "  "; // Spacer
    os << "g_" << metric.Dgi[i1] << metric.Dgj[i1] << ',' << metric.Dgk[i1] 
       << " = " << std::setw(10) 
       << metric.Dg(i1);
  }
  os << std::endl;

  // Output g^ij,_k
  for (int i1=0;i1<metric.NDg;i1++){
    if (i1)
      os << "  "; // Spacer
    os << "g^" << metric.Dgi[i1] << metric.Dgj[i1] << ",_" << metric.Dgk[i1] 
       << " = " << std::setw(10) 
       << metric.Dginv(i1);
  }
  os << std::endl;
  
  std::cerr << "at x = (" << metric._x[0] << ", " << metric._x[1]
	    << ", " << metric._x[2] << ", " << metric._x[3] << ')' << std::endl;

  return os;
}

};
