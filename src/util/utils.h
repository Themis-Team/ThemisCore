/*!
  \file utils.h
  \author Avery E. Broderick
  \date  April, 2017
  \brief Header file for a variety of utility functions within Themis.
*/

#ifndef Themis_UTILS_H_
#define Themis_UTILS_H_

#include <string>
#include <vector>
#include <complex>
#include <sys/stat.h>


#ifndef THEMISPATH
#define THEMISPATH ("./")
#endif


namespace Themis {

/*!
  \brief Defines a set of general utility functions.

  \details Provides implementations of utility functions that centralizes
   this functionality for Themis.
*/
  namespace utils
  {

    //! Checks whether a file exists
    bool isfile(const std::string& name); 
  
    //! Generates the time in seconds since 00:00, Jan 1, 2000. 
    //! Takes year, days since Jan 1, and hour in 2400 clock.
    double time_J2000(int year, int day, int hour, int min, double sec);

    //! Strips off and returns file extension
    std::string get_file_extension(std::string file_name);

    //! Prepends global path to the Themis directory tree to file names
    //! to give global behavior while permiting relative naming conventions
    std::string global_path(std::string file_name);

    //! Returns a vector of station codes.  Takes a name, which currently
    //! is "Two-letter 2017" or "HOPS 2017" or "uvfits 2017"
    std::vector<std::string> station_codes(std::string listname="Two-letter 2017");

    //! Fast exp for purely imaginary arguments good to 0.00013%
    //! Accepts arguments in fraction of full phase, i.e., x=1
    //! corresponds to a phase of 2pi.
    //! Assumes that the argument is within +-MAX_INT
    static inline std::complex<double> fast_img_exp7(double x)
    {
      // Four-term polynomial coefficients for sine on [-pi/2,pi/2].
      // Computed by fixing value and derivatives at 0 and pi/2 and
      // minimizing the L2 distance from sin(x) on the range.
      // Not that this will rescale the argument so that this maps to
      // [-1,1]. 

      static double sa=1.5707963267948966192, sb=-0.64590160249678916295, sc=0.079414224608888468197, sd=-0.0043089489069959244829;

      static int sine_shift[8] = {0,2,2,-4,-4,6,6,-8};
      static int sine_sign[8] = {1,-1,-1,1,1,-1,-1,1};

      static int cosine_shift[8] = {1,1,-3,-3,5,5,-7,-7};
      static int cosine_sign[8] = {-1,-1,1,1,-1,-1,1,1};
  
      // Quick put into one phase between [0,8], two periods
      x = 4*(x-int(x)+1);

      // Select correct quarter phase
      int qps = int(x);
      double xs = sine_sign[qps]*x + sine_shift[qps];
      double xc = cosine_sign[qps]*x + cosine_shift[qps];
  
      double s = xs* ( sa + xs*xs*( sb + xs*xs*( sc + xs*xs*sd )));
      double c = xc* ( sa + xc*xc*( sb + xc*xc*( sc + xc*xc*sd )));

      double norm = 1.5-0.5*(s*s+c*c);
  
      return std::complex<double>( c*norm, s*norm );
    }

    //! Fast exp for purely imaginary arguments good to 0.008%
    //! Accepts arguments in fraction of full phase, i.e., x=1
    //! corresponds to a phase of 2pi.
    //! Assumes that the argument is within +-MAX_INT    
    static inline std::complex<double> fast_img_exp5(double x)
    {
      // Three-term polynomial coefficients for sine on [-pi/2,pi/2].
      // Computed by fixing value and derivatives at 0 and pi/2.
      // Not that this will rescale the argument so that this maps to
      // [-1,1]. 
      // static double sa=0.5*M_PI, sb=2.5-M_PI, sc=-1.5+0.5*M_PI;
      static double sa=1.5707963268, sb=-0.6415926536, sc=0.07079632679;
      
      // Define a set of mappings of sine from [-2pi,2pi]
      //   to sine on [-pi/2,pi/2].
      static int sine_shift[8] = {0,2,2,-4,-4,6,6,-8};
      static int sine_sign[8] = {1,-1,-1,1,1,-1,-1,1};

      // Define a set of mappings of cosine from [-2pi,2pi]
      //   to sine on [-pi/2,pi/2].
      static int cosine_shift[8] = {1,1,-3,-3,5,5,-7,-7};
      static int cosine_sign[8] = {-1,-1,1,1,-1,-1,1,1};
  
      // Quick put into one phase between [0,8], two periods
      x = 4*(x-int(x)+1);

      // Select correct quarter phase
      int qps = int(x);
      double xs = sine_sign[qps]*x + sine_shift[qps];
      double xc = cosine_sign[qps]*x + cosine_shift[qps];

      // Apply the three-term approximate
      double s = xs* ( sa + xs*xs*( sb + xs*xs*sc ));
      double c = xc* ( sa + xc*xc*( sb + xc*xc*sc ));

      // Renormalize assuming small correction
      double norm = 1.5-0.5*(s*s+c*c);

      // Create and return complex number.
      return std::complex<double>( c*norm, s*norm );
    };
      
  };
  
};

#endif
