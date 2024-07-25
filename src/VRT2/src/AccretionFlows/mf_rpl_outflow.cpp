#include "mf_rpl_outflow.h"

namespace VRT2 {

MF_RPL_Outflow::MF_RPL_Outflow(Metric& g, AccretionFlowVelocity& u, double BP0, double BPindex, double OmegaF)
  : MagneticField(g), _u(u), _BP0(BP0), _BPindex(BPindex)
{
  _OmegaF = OmegaF * (_g.ang_mom()/_g.mass()) / ( 2.0 * _g.horizon() );
  //_OmegaF = OmegaF;
}

FourVector<double>& MF_RPL_Outflow::get_field_fourvector(double t, double r, double theta, double phi)
{
  FourVector<double> utmp = _u(t,r,theta,phi);

  // Define time components of the Maxwellian
  double sFtr = (std::cos(theta)<0 ? -1 : 1 ) * _BP0 * std::pow(r,_BPindex); // *F^{tr} ( sign ensures that field enters below and exits above)
  double sFtth = utmp.con(2) * sFtr / utmp.con(1);                           // *F^{ttheta}
  double sFtph = ( utmp.con(3)-_OmegaF*utmp.con(0) ) * sFtr / utmp.con(1);   // *F^{tph}

  // Define something that is not quite a four-vector, but we want to do a 
  //  four-vector multiplication with it (called B in McKinney papers)
  //  (Note that this is a four-vector if it is defined by B=*F.n, where n_mu = (1,0,0,0))
  _b.mkcon(0.0,sFtr,sFtth,sFtph);
  
  // Then use b^mu = ( 1^mu_nu + u^mu u_nu ) *F^{tnu} / u^t to get true B-field four-vector
  _b += (utmp*_b) * utmp;
  _b *= (1.0/utmp.con(0));

  return _b;
}

};
