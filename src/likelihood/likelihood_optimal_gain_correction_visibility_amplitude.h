/*! 
  \file likelihood_optimal_gain_correction_visibility_amplitude.h
  \author Avery E. Broderick
  \date  October, 2018
  \brief Header file for the likelihood_optimal_gain_correction_visibility_amplitude likelihood class
  \details Derived from the base likelihood class. Returns the natural log of 
  the likelihood. The data and model objects are passed to the constructor.
*/

#ifndef THEMIS_LIKELIHOOD_OPTIMAL_GAIN_CORRECTION_VISIBILITY_AMPLITUDE_H_
#define THEMIS_LIKELIHOOD_OPTIMAL_GAIN_CORRECTION_VISIBILITY_AMPLITUDE_H_

#include <vector>
#include <string>

#include "likelihood_base.h"
#include "data_visibility_amplitude.h"
#include "model_visibility_amplitude.h"
#include <mpi.h>

namespace Themis
{

  /*! 
    \brief Defines a likelihood that non-linearly optimizes over gain corrections with a Gaussian prior.
    
    \details This class takes a visibility amplitude data object and a visibility amplitude model object, 
    and then returns the log likelihood after minimalization of the log-likelihood within the gain-correction
    sub-space.  That is, it sets the model to \f$ V_{AB}(g_A,g_B;p) = (1+g_A)(1+g_B) \bar{V}_{AB}(p) \f$,
    and numerically maximizes the likelihood independently for each observation epoch, assuming Gaussian priors
    on the \f$g_A\f$ and Gaussian errors.  An approximation of the marginalized likelihood is generated using 
    the covariance of the likelihood, though this makes little difference in practice. Gain-reconstruction epochs
    may be specified explicitly; by default the corrections will be performed by scan.

    \warning   Note that this fails if low-SNR points are included in the analysis, which drives erroneous gain reconstructions.  Typically, a minimum SNR or 2 is sufficient, which is required for the visibility amplitude error distribution to be well approximated by a Gaussian.
  */
  class likelihood_optimal_gain_correction_visibility_amplitude : public likelihood_base
  {
  public:

    //! Construct to do scan-by-scan correction by default
    likelihood_optimal_gain_correction_visibility_amplitude(data_visibility_amplitude& data, model_visibility_amplitude& model, std::vector<std::string> station_codes, std::vector<double> sigma_g);

    //! Specify the times of the gain-correction epochs by hand
    likelihood_optimal_gain_correction_visibility_amplitude(data_visibility_amplitude& data, model_visibility_amplitude& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge);

    //! Specify the times of the gain-correction epochs by hand
    likelihood_optimal_gain_correction_visibility_amplitude(data_visibility_amplitude& data, model_visibility_amplitude& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g);

     
    ~likelihood_optimal_gain_correction_visibility_amplitude();

    //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    virtual double operator()(std::vector<double>& x);
     
    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
    virtual double chi_squared(std::vector<double>& x);

    //! Returns the start time of each gain-correction epoch.
    std::vector<double>  get_gain_correction_times();
    //! Returns the set of gain corrections for each epoch.
    std::vector< std::vector<double> >  get_gain_corrections();
    //! Outputs the set of gain corrections in a structure fashion to an output stream
    void output_gain_corrections(std::ostream& out);
    //! Outputs the set of gain corrections in a structure fashion to a file
    void output_gain_corrections(std::string outname);
	
          
    //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
    virtual void set_mpi_communicator(MPI_Comm comm);

    //! Returns the number of independent gains, which is generally less than product of the number of stations and number of epochs.
    size_t number_of_independent_gains();
    
    //! Reads gain file
    void read_gain_file(std::string gain_file_name);

    //! Turns on gain solver (default is on)
    void solve_for_gains();

    //! Fixes the gains to last value.
    void fix_gains();
    
    //! Assume that the gain corrections are correlated, yielding an approximate solution for subsequent times (default).
    void assume_smoothly_varying_gains();

    //! Assume that the gain corrections are not correlated
    void assume_independently_varying_gains();

    
  protected:

    //! Outputs the data and model, as modified by the likelihood appropriately,
    //! to the specified output stream.  Useful for comparison later.
    virtual void output(std::ostream& out);

    
  private:
     data_visibility_amplitude& _data;
     model_visibility_amplitude& _model;  
     
     std::vector<std::string> _station_codes;
     std::vector<double> _sigma_g, _max_g;

     std::vector<double> _tge;
     std::vector< std::vector<double> > _g;

     std::vector<double> _sqrt_detC;
     
     std::vector<double> _x; // Parameter list

     bool _smoothly_varying_gains;

     bool _solve_for_gains;

     void check_station_codes();
     void allocate_memory();


     void organize_data_lists();
     
     std::vector< std::vector<size_t> > _datum_index_list;
     std::vector< std::vector<double> > _y_list;
     std::vector< std::vector<size_t> > _is1_list;
     std::vector< std::vector<size_t> > _is2_list;
     

     double optimal_gain_corrections(std::vector<double>& y, std::vector<double>& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector<double>& gest);

     // Compute the inverse in place and the determininate of a matrix
     double matrix_determinant(double** a);
     // Matrix determinant
     void ludcmp(double **a, int n, int *indx, double &d);
     
     // Levenberg-Marquardt Method
     // Assumes that _y, _yb, _is1, _is2 are set
     double *_ogc_y, *_ogc_yb;
     size_t *_ogc_is1, *_ogc_is2;
     void gain_optimization_likelihood(size_t i, const double g[], double *y, double dydg[]) const;
     
     void covsrt(double **covar, int ma, int mfit);
     void gaussj(double **a, int n, double **b, int m);
     void mrqcof(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq);

     double _mrq_ochisq, *_mrq_atry, *_mrq_beta, *_mrq_da, **_mrq_oneda;
     
     void mrqmin(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda);
  };
  
};

#endif 
