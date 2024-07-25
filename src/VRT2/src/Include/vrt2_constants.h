#ifndef VRT2_VRT2_CONSTANTS_H
#define VRT2_VRT2_CONSTANTS_H

namespace VRT2 {
namespace VRT2_Constants
{
  /*** Fundamental Constants ***/
  static const double pi = 3.14159265358979323846; // pi
  static const double G = 6.67259E-8;              // Newton's constant
  static const double e = 4.8032E-10;              // electron charge in esu
  static const double me = 9.10938188E-28;         // electron mass in g
  static const double mp = 1.6726e-24;             // proton mass in g
  static const double c = 2.99792458e10;           // speed of light in vacuum in cm/s
  static const double hbar = 1.054571596E-27;      // reduced Planck's constant in erg s
  static const double re = 2.817940285e-13;        // classical electron radius in cm (e^2 / me c^2)
  static const double k = 1.3806503e-16;           // Boltzmann constant
  static const double sigma = 5.67051e-5;          // Stefan-Boltzmann constant

  /*** Fiducial numbers ***/
  static const double M_sun = 1.9889e33;           // Mass of sun in g


  /*** Derived Constants ***/
  static const double Cwp2 = 4.0*pi*e*e/(me*c*c);  // wp^2 = Cwp2 * n where n is in cm^-3 and wp is in cm^-1
  static const double CwB = e/(me*c*c);            // wB = CwB * B where B is in Gauss and wB is in cm^-1
  static const double CwB2 = CwB*CwB;              // wB^2 = CwB2 * B^2  "

  /*** Useful Parameters ***/
  static const double M_SgrA_sun = 4.30e6;         // Mass of Sgr A* in solar masses (Gillessen et al., 2009, ApJ 707 L114)
  static const double M_SgrA_g = M_SgrA_sun * M_sun; // Mass of Sgr A* in g
  static const double M_SgrA_cm = M_SgrA_sun * G*M_sun/(c*c); // Mass of Sgr A* in cm

  //static const double D_SgrA_kpc = 8.3;           // Distance to Sgr A* in kpc (Gillessen et al., 2009, ApJ 707 L114)
  static const double D_SgrA_kpc = 8.122;           // Distance to Sgr A* in kpc LAST FIT
  static const double D_SgrA_cm = D_SgrA_kpc * 1e3 * 3.086e18; // Distance to Sgr A* in cm



  //conversion ratios
  static const double solarmass_to_cm = G*M_sun/(c*c);		//solar mass to cm conversion (BH mass)
  static const double kpc_to_cm = 1e3 * 3.086e18;			//kpc to cm conversion (BH diameter)

  /** For M87 **/
  //static const double M_M87_sun = 3.4e9;  // Mass of M87 in solar masses (Old value!)
  static const double M_M87_sun = 6.5e9;  // Mass of M87 in solar masses (EHT Paper VI)
  static const double M_M87_g = M_M87_sun * M_sun;
  static const double M_M87_cm = M_M87_sun * G*M_sun/(c*c);

  static const double D_M87_Mpc = 16.9;           // Distance to M87 in Mpc used in Paper I EHT 
  static const double D_M87_cm = D_M87_Mpc * 1e6 * 3.086e18; // Distance to M87 in cm
  
  static const double D_M87_pc_2017 = 16.9e6;  // Assumed distance in pc from Gebhardt et al.,  2011, ApJ 729 119, mass scales linearly with this.
  static const double D_M87_cm_2017 = D_M87_pc_2017 * 3.086e18;
  
  /** For 3c454.3 **/
  static const double M_3c454_sun = 1.0e9; // Approx. mass of 3c454.3 SMBH in solar masses (Jorstad et al., 2013 ApJ 773 147)
  static const double M_3c454_g = M_3c454_sun * M_sun;
  static const double M_3c454_cm = M_3c454_sun * G*M_sun/(c*c);

  static const double D_3c454_pc = 1544.0e6; // Ang. Size Distance in pc (from NEDS)
  static const double D_3c454_cm = D_3c454_pc * 3.086e18;



};
};
#endif


