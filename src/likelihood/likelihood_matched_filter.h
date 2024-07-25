/*!
  \file likelihood_matched_filter.h
  \author Paul Tiede
  \date  Jan, 2019
  \brief Header file for the matched filter likelihood class
*/

#ifndef THEMIS_LIKELIHOOD_MATCHED_FILTER_H_
#define THEMIS_LIKELIHOOD_MATCHED_FILTER_H_
#include <vector>
#include <cmath>
#include <iostream>
#include "likelihood_base.h"
#include <mpi.h>
#include <fstream>
#include <istream>
#include "constants.h"
namespace Themis{

  /*!
    \brief Defines a class that constructs a flux likelihood object
    
    \details Does matched filtering with a slashed Gaussian elliptical ring to extract widths position and diameters.
             -parameter list:
               parameter[0] = semi-major axis diameter in uas
               parameter[1] = ring width (FWHM of ring) in uas
               parameter[2] = ring eccentricity (-1 to 1) in uas
               parameter[3] = rotation angle measured from north of east (must go from 0 to 2*pi due to the branch cut I picked)
               parameter[4] = ring slash [0,1], where 0 is no slash and 1 means brightness drops to zero.
               parameter[5] = x ring center in uas
               parameter[6] = y ring center in uas
  */
  class likelihood_matched_filter:public likelihood_base
  {
    public:
      likelihood_matched_filter(double lreg, double treg, double I_clip, std::string image_fname, std::string READ_fname, std::string image_type);
                                      
      ~likelihood_matched_filter() {};
    
      virtual double operator()(std::vector<double>& x);
      virtual double chi_squared(std::vector<double>& x);
    
      //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
      virtual void set_mpi_communicator(MPI_Comm comm);
      
    protected:

      //! Outputs the data and model, as modified by the likelihood appropriately,
      //! to the specified output stream.  Useful for comparison later.
      virtual void output(std::ostream& out);
  	
    private:
      //! multiplicative factor in front of the log likelihood that controls the the specificity of the likelihood. Often lreg=1 fails to properly contrain the fit.
      const double _lreg;
      const double _treg;
      const double _I_clip;
      std::vector<std::vector<double> > _alpha, _beta, _I;
      //! Finds the intensity (normalized so total flux is unity) of a Gaussian ring according to the images alpha and beta
      std::vector<std::vector<double> > gaussian_ring_model(std::vector<double> parameters);

      void GRMHD_read(std::string image_fname, std::string README_fname);
      void test_read();
      void SMILI_read(std::string image_fname, std::string README_fname);
      void hpu_read(std::string image_fname);
      
      //Finds the minimum distance squared between x,y and an ellipse. Uses an iterative method from
      // https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
      double find_min_distance_square(double a, double b, double x, double y, double eps=1e-8, int nmax=20);
  };
};

#endif 
