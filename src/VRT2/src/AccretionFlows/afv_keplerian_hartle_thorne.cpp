#include "afv_keplerian_hartle_thorne.h"

namespace VRT2 {
AFV_Keplerian_HartleThorne::AFV_Keplerian_HartleThorne(Metric& g, double ri)
  : AccretionFlowVelocity(g), _ri(ri)
{

  _N = 1000;
  _lr = new double[_N];
  _Omega = new double[_N];
  double rmin = 1.001*_g.horizon();
  double rmax = 1e5;
  for (int i=0; i<_N; ++i)
  {
    _lr[i] = std::log(rmin) + std::log(rmax/rmin)*i/double(_N-1);
    _Omega[i] = Omega(std::exp(_lr[i]));

    
  }
  _lrmin = std::log(rmin);
  _lrmax = std::log(rmax);


  // Get local position
  std::valarray<double> x = _g.local_position();
  _g.reset(0,_ri,0.5*M_PI,0);
  
  // Set angular momentum and energy cutoffs
  _u = get_keplerian_velocity(_ri);
  _Omegai = _u.con(3)/_u.con(0);

  // return metric to where it was
  _g.reset(x);


  /*
  std::ofstream out("AFV_Keplerian_HartleThorne-Omega.d");
  for (int j=0; j<_N; ++j)
  {
    out << std::setw(15) << std::exp(_lr[j])
	<< std::setw(15) << _Omega[j]
	<< std::endl;
  }
  out << "\n\n" << std::endl;

  for (int j=0; j<100; ++j)
  {
    double rtmp = _ri + 15.0/99.0 * j;
    double lr = std::log(rtmp);
    int i = int( (lr-_lrmin)/(_lrmax-_lrmin) * _N );
    double Omega = 0;
    if (i>=0  && i<(_N-1))
    {
      double dlr = (lr-_lr[i])/(_lr[i+1]-_lr[i]);
      Omega = _Omega[i]*(1-dlr) + _Omega[i+1]*dlr;
    }
    out << std::setw(15) << rtmp
	<< std::setw(15) << Omega
	<< std::endl;
      

  }
  out << "\n\n" << std::endl;

  out << std::setw(15) << _ri
      << std::setw(15) << _Omegai
      << std::endl;
  out.close();
  std::exit(0);
  */
}
AFV_Keplerian_HartleThorne::~AFV_Keplerian_HartleThorne()
{
  delete[] _lr;
  delete[] _Omega;
}

double AFV_Keplerian_HartleThorne::Omega(double r)
{
  // Get local position
  std::valarray<double> x = _g.local_position();
  _g.reset(0,r,0.5*M_PI,0);
  

  double gttr = _g.Dginv(0,0,1);
  double gtpr = _g.Dginv(0,3,1);
  double gppr = _g.Dginv(3,3,1);
  double l = - gtpr/gppr - std::sqrt( std::pow(gtpr/gppr,2) - gttr/gppr );
  
  double gtt = _g.ginv(0,0);
  double gtp = _g.ginv(0,3);
  double gpp = _g.ginv(3,3);
  double o = (gpp*l + gtp)/(gtt + gtp*l);

  // return metric to where it was
  _g.reset(x);  

  return o;
}
};
