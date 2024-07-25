/***********************************************************************/
/*** Defines a power-law magnetic field for an outflow               ***/
/*                                                                     */
/* Assumes stationarity, azimuthal symmetry.  Does this via            */
/* flux conservation laws along field lines, which requires the        */
/* velocity field:                                                     */
/*                                                                     */
// u^r/*F^{tr} = u^theta/*F^{ttheta} = (u^phi-u^t Omega_F)/*F^{tphi}   */
/*                                                                     */
/* The radial field is taken to be a power-law.  The field angular     */
/* frequency is defined in units of the black hole angular frequency   */
/* which is defined as j                                               */
/*                                                                     */
/* ASSUMES THAT ur IS NON-ZERO ---> CAN USE *F^{tr} TO DEFINE b        */
/*                                                                     */
/* ASSUMES THAT g HAS ALREADY BEEN RESET                               */
/*                                                                     */
/***********************************************************************/

#ifndef VRT2_MF_RPL_OUTFLOW_H
#define VRT2_MF_RPL_OUTFLOW_H

#include <cmath>
#include <math.h>
using namespace std;
#include "vrt2_constants.h"
#include "magnetic_field.h"
#include "accretion_flow_velocity.h"

namespace VRT2 {
class MF_RPL_Outflow : public MagneticField
{
 public:
  MF_RPL_Outflow(Metric& g, AccretionFlowVelocity& u, double BP0, double BPindex, double OmegaF=0.5);
  virtual ~MF_RPL_Outflow() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);

 protected:
  AccretionFlowVelocity& _u;
  double _BP0, _BPindex;
  double _OmegaF;
};
};
#endif
