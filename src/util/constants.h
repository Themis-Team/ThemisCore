/*!
  \file constants.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Provides a list of constants (in cgs) to be used throughout Themis.
  \details To be added
*/

#ifndef Themis_CONSTANTS_H_
#define Themis_CONSTANTS_H_

namespace Themis {

  /*!
    \brief A variety of constants (in cgs) within the namespace Themis::constants
     to be used throughout Themis.

    \details More details to be added

    \todo Add additional physical and astronomical constants.
  */
  namespace constants
  {
    static const double G = 6.67259e-8;         //!< Newton's constant
    static const double e = 4.8032e-10;         //!< Electron charge in esu
    static const double me = 9.10938188e-28;    //!< Electron mass in g
    static const double mp = 1.6726e-24;        //!< Proton mass in g
    static const double c = 2.99792458e10;      //!< Speed of light in vacuum in cm/s
    static const double h = 6.62606885e-27;     //!< Planck's constant in erg s
    static const double hbar = 1.054571596e-27; //!< Reduced Planck's constant in erg s
    static const double re = 2.817940285e-13;   //!< Classical electron radius in cm (e^2 / me c^2)
    static const double k = 1.3806503e-16;      //!< Boltzmann constant
    static const double sigma = 5.67051e-5;     //!< Stefan-Boltzmann constant
    static const double Msun = 1.98855e33;      //!< Mass of sun in g
    static const double pc = 3.086e18;          //!< Parsec in cm
    static const double AU = 1.495978707e13;    //!< Astronomical unit in cm
  };

};

#endif
