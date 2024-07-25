/************************************************************/
/*** Defines the magnetic field from the Falcke jet model ***/
/*                                                          */
/* NOTE THAT THIS EXPECTS THAT THE METRIC                   */
/* AND OTHER SUPPLIED ITEMS HAVE BEEN RESET                 */
/* TO THE CURRENT POSITION.                                 */
/*                                                          */
/************************************************************/

#ifndef VRT2_MF_FALCKE_H
#define VRT2_MF_FALCKE_H

#include "vrt2_constants.h"
#include "magnetic_field.h"
#include "Falcke_jet_model.h"
#include "accretion_flow_velocity.h"

namespace VRT2 {
class MF_Falcke : public MagneticField
{
 public:
  MF_Falcke(Metric& g, double B0, FalckeJetModel& jet, AccretionFlowVelocity& afv);
  virtual ~MF_Falcke() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);

 protected:
  double _B0;
  FalckeJetModel& _jet;
  AccretionFlowVelocity& _afv;
};

};
#endif
