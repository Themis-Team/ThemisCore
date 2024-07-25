#include "rt_ht_check.h"

namespace VRT2 {
RT_HartleThorneCheck::RT_HartleThorneCheck(Metric& g, double M, double a)
 : RadiativeTransfer(g), _gk(M,a)
{
}
RT_HartleThorneCheck::RT_HartleThorneCheck(const double y[], Metric& g, double M, double a)
 : RadiativeTransfer(y,g), _gk(M,a)
{
}
RT_HartleThorneCheck::RT_HartleThorneCheck(FourVector<double>& x, FourVector<double>& k, Metric& g, double M, double a)
 : RadiativeTransfer(x,k,g), _gk(M,a)
{
}

double RT_HartleThorneCheck::ht_fdiff(double y[])
{
  _gk.reset(y);
  _g.reset(y);
  Metric& gk=_gk;
  
  double diff = 0.0;
  for (int i=0; i<4; ++i)
    for (int j=0; j<4; ++j)
      {
	diff += (std::fabs(gk.g(i,j))>0 ? std::pow( (gk.g(i,j) - _g.g(i,j))/gk.g(i,j) , 2.0)  : 0);
      }

  return std::sqrt(diff);
}

std::valarray<double>& RT_HartleThorneCheck::IQUV_integrate(std::vector<double> y[],std::vector<double> dydx[],std::valarray<double>& iquv)
{
  double diffmax = 0;
  double ytmp[4];
  for (size_t i=0; i<y[0].size(); i++)
  {
    for (int j=0; j<4; ++j)
      ytmp[j] = y[j+1][i];
    diffmax = std::max(diffmax,ht_fdiff(ytmp));
  }

  iquv[0] = diffmax;
  iquv[1]=iquv[2]=iquv[3] = 0.0;



  return iquv;
}
};
