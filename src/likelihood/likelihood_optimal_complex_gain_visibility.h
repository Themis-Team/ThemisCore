/*! 
  \file likelihood_optimal_complex_gain_visibility.h
  \author Avery E. Broderick
  \date  February, 2020
  \brief Header file for the likelihood_optimal_complex_gain_visibility likelihood class
  \details Derived from the base likelihood class. Returns the natural log of 
  the likelihood. The data and model objects are passed to the constructor.
*/

#ifndef THEMIS_LIKELIHOOD_OPTIMAL_COMPLEX_GAIN_VISIBILITY_H_
#define THEMIS_LIKELIHOOD_OPTIMAL_COMPLEX_GAIN_VISIBILITY_H_

#include <vector>
#include <string>
#include <complex>

#include "likelihood_base.h"
#include "data_visibility.h"
#include "model_visibility.h"
#include "uncertainty_visibility.h"

#include <mpi.h>

namespace Themis
{

  /*! 
    \brief Defines a likelihood that non-linearly optimizes over gain corrections with a Gaussian prior.
    
    \details This class takes a visibility data object and a visibility model object, and then returns the 
    log likelihood after minimalization of the log-likelihood within the complex gain-correction
    sub-space.  That is, it sets the model to \f$ V_{AB}(G_A,G_B;p) = G_A G_B \bar{V}_{AB}(p) \f$,
    and numerically maximizes the likelihood independently for each observation epoch, assuming Gaussian priors
    on the \f$|G_A|\f$ and Gaussian errors.  An approximation of the marginalized likelihood is generated using 
    the covariance of the likelihood, though this makes little difference in practice. Gain-reconstruction epochs
    may be specified explicitly; by default the corrections will be performed by scan.

    \warning
  */
  class likelihood_optimal_complex_gain_visibility : public likelihood_base
  {
  public:

    //! Construct to do scan-by-scan correction by default
    likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, std::vector<std::string> station_codes, std::vector<double> sigma_g);

    //! Specify the times of the gain-correction epochs by hand
    likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge);

    //! Specify the times of the gain-correction epochs by hand
    likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g);

    //! Construct to do scan-by-scan correction by default
    likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, uncertainty_visibility& visibility, std::vector<std::string> station_codes, std::vector<double> sigma_g);

    //! Specify the times of the gain-correction epochs by hand
    likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, uncertainty_visibility& visibility, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge);

    //! Specify the times of the gain-correction epochs by hand
    likelihood_optimal_complex_gain_visibility(data_visibility& data, model_visibility& model, uncertainty_visibility& visibility, std::vector<std::string> station_codes, std::vector<double> sigma_g, std::vector<double> t_ge, std::vector<double> max_g);
    
     
    ~likelihood_optimal_complex_gain_visibility();

    //! Returns the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    virtual double operator()(std::vector<double>& x);

    //! Returns the gradient of the log-likelihood of a vector of parameters \f$ \mathbf{x} \f$
    //! The prior permits parameter checking if required during likelihood gradient evaluation, through the gradients of the prior is applied elsewhere.
    virtual std::vector<double> gradient(std::vector<double>& x, prior& Pr);
     
    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$
    virtual double chi_squared(std::vector<double>& x);

    //! Returns the \f$ \chi^2 \f$ of a vector of parameters \f$ \mathbf{x} \f$ including the gain prior terms
    virtual double chi_squared_with_gain_priors(std::vector<double>& x);
    
    //! Returns the start time of each gain-correction epoch.
    std::vector<double>  get_gain_times();
    //! Returns the set of gain corrections for each epoch.
    std::vector< std::vector< std::complex<double> > >  get_gains();
    //! Outputs the set of complex gains in a structure fashion to an output stream
    void output_gains(std::ostream& out);
    //! Outputs the set of complex gains in a structure fashion to a file
    void output_gains(std::string outname);
    //! Outputs the set of gain corrections in a structure fashion to an output stream
    void output_gain_corrections(std::ostream& out);
    //! Outputs the set of gain corrections in a structure fashion to a file
    void output_gain_corrections(std::string outname);
	
          
    //! Defines a set of processors provided to the model for parallel computation via an MPI communicator.  Only facilates code parallelization if the model computation is parallelized via MPI.
    virtual void set_mpi_communicator(MPI_Comm comm);

    //! Returns the number of independent gains, which is generally less than product of the number of stations and number of epochs.
    size_t number_of_independent_gains();

    //! Set the maximum number of iterations in the likelihood maximizer.  More iterations results in higher accuracy for the gains, but at the expense of more time.
    void set_iteration_limit(int itermax);

    //! Reads gain file
    void read_gain_file(std::string gain_file_name);

    //! Turns on gain solver (default is on)
    void solve_for_gains();

    //! Fixes the gains to last value.
    void fix_gains();
    
    //! Fixes the gains during gradient evaluation
    void fix_gains_during_gradient();
    
    //! Fixes the gains during gradient evaluation
    void solve_for_gains_during_gradient();

    //! Assume that the previous solutions for the complex gains yield an approximate solution for the next iteration.  Note that this does make subsequent gain solves dependent on the prior history.  However, it can also make the solution of the gains much faster.  It overrides assume_smoothly_varying_gains() and assume_independently_varying gains().
    void use_prior_gain_solutions();
        
    //! Assume that the gain corrections are correlated, yielding an approximate solution for subsequent times (default).
    void assume_smoothly_varying_gains();

    //! Assume that the gain corrections are not correlated
    void assume_independently_varying_gains();
    
  protected:

    //! Outputs the data and model, as modified by the likelihood appropriately,
    //! to the specified output stream.  Useful for comparison later.
    virtual void output(std::ostream& out);

    
  private:
     data_visibility& _data;
     model_visibility& _model;  
     uncertainty_visibility _local_uncertainty; // Default if none is passed
     uncertainty_visibility& _uncertainty;
     
     std::vector<std::string> _station_codes;
     std::vector<double> _sigma_g, _max_g;

     std::vector<double> _tge;
     std::vector< std::vector< std::complex<double> > > _G;

     std::vector<double> _sqrt_detC;
     
     bool _use_prior_gain_solutions;
     bool _smoothly_varying_gains;

     bool _solve_for_gains;
     bool _solve_for_gains_during_gradient;
     
     void check_station_codes();
     void allocate_memory();

     bool _parallelize_likelihood;
     double likelihood_multiproc(std::vector<double>& x);
     double likelihood_uniproc(std::vector<double>& x);

     void distribute_gains();

     
     void organize_data_lists();
     
     std::vector< std::vector<size_t> > _datum_index_list;
     std::vector< std::vector< std::complex<double> > > _y_list;
     std::vector< std::vector<size_t> > _is1_list;
     std::vector< std::vector<size_t> > _is2_list;
     
     double _opi2; // Prior on phase to force unique solutions.  Should be very large (1e-4).

     int _itermax; // Maximum number of iterations

     std::vector<std::complex<double> > generate_gain_guess(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, int repeats=5);
     
     double optimal_complex_gains(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest);

     double optimal_complex_gains_trial(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector< std::complex<double> >& gest, double& chisq);

     double optimal_gain_amplitude_trial(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector<std::complex<double> >& gest, double& chisq_opt);

     double optimal_complex_gains_log_trial(std::vector< std::complex<double> >& y, std::vector< std::complex<double> >& yb, std::vector<size_t>& is1, std::vector<size_t>& is2, std::vector<std::complex<double> >& gest, double& chisq_opt);
     
     // Compute the inverse in place and the determininate of a matrix
     int *_indx, *_indxc, *_indxr, *_ipiv;
     double *_vv, *_dyda;
     double matrix_determinant(double** a);
     // Matrix determinant
     void ludcmp(double **a, int n, int *indx, double &d);
     
     // Levenberg-Marquardt Method
     // Assumes that _y, _yb, _is1, _is2 are set
     double *_ogc_y, *_ogc_yb, *_sig;
     size_t *_ogc_is1, *_ogc_is2;
     double **_covar, **_alpha, *_g, *_og;
     void gain_optimization_likelihood(size_t i, const double g[], double *y, double dydg[]) const;
     void gain_amplitude_optimization_likelihood(size_t i, const double g[], double *y, double dydg[]) const;
     void gain_optimization_log_likelihood(size_t i, const double g[], double *y, double dydg[]) const;
     
     void covsrt(double **covar, int ma, int mfit);
     int gaussj(double **a, int n, double **b, int m);
     void mrqcof(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq);
     void mrqcof_amplitude(double y[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq);
     void mrqcof_log(double y[], double sig[], int ndata, double a[], int ma, double **alpha, double beta[], double *chisq);
  
     double _mrq_ochisq, *_mrq_atry, *_mrq_beta, *_mrq_da, **_mrq_oneda;
     
     int mrqmin(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda);
     int mrqmin_amplitude(double y[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda);
     int mrqmin_log(double y[], double sig[], int ndata, double a[], int ma, double **covar, double **alpha, double *chisq, double *alamda);


     std::vector<double> _x_debug;

     std::vector<double> _x_last;
     double _L_last;
  };
  
};

#endif 
