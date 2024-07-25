#include "Falcke_jet_model.h"

namespace VRT2 {

FalckeJetModel::FalckeJetModel(double Znozzle, double adiabatic_index)
  : _Znozzle(Znozzle), _adiabatic_index(adiabatic_index)
{
  _bs = std::sqrt( (_adiabatic_index-1)/(_adiabatic_index+1) );
  _gbs = _bs/std::sqrt(1-_bs*_bs);
}

double FalckeJetModel::gbj(double z)
{

  double Heinos_number = 1.0/0.87; // Magic number from fit
  double gb;
  z = std::max(std::fabs(z),_Znozzle);
  gb = Heinos_number * _bs * std::sqrt( 4.0*std::log(std::fabs(z/_Znozzle)) + _adiabatic_index );

  //gb = 2.0;  // Blandford - Konigle test.

  return gb;
}

double FalckeJetModel::mach(double z)
{
  return gbj(z)/_gbs;
}

double FalckeJetModel::rperp(double z)
{
  z = std::max(std::fabs(z),_Znozzle);

  return z/mach(z);
}

};
