/*!
	\file model_image_slashed_ring.cpp
	\author Jorge A. Preciado
	\date  October 2017
	\brief Implements Slashed Ring Model Image class.
	\details To be added
*/

#include "model_image_slashed_ring.h"
#include <iostream>
#include <iomanip>
#include <complex>
#include <mpi.h>

namespace Themis {

	model_image_slashed_ring::model_image_slashed_ring()
		: _Nray(128), _use_analytical_visibilities(true)
	{
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		std::cout << "Creating model_image_slashed_ring in rank " << world_rank << std::endl;
	}

	void model_image_slashed_ring::use_numerical_visibilities()
	{
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		std::cout << "Using numerical visibilities in rank " << world_rank << std::endl;

		_use_analytical_visibilities = false;
	}
	
	void model_image_slashed_ring::use_analytical_visibilities()
	{
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		std::cout << "Using analytical visibilities in rank " << world_rank << std::endl;
		
		_use_analytical_visibilities = true;
	}

	void model_image_slashed_ring::set_image_resolution(int Nray)
	{
		_Nray = Nray;
		
		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		std::cout << "model_image_slashed_ring: Rank " << world_rank << " using image resolution " << _Nray << std::endl;
	}
	

	void model_image_slashed_ring::generate_image(std::vector<double> parameters, std::vector<std::vector<double> >& I, std::vector<std::vector<double> >& alpha, std::vector<std::vector<double> >& beta)
	{
		int Ntot = _Nray;
		double Rs = 2.0; // Number of Radius
		
		_V0 = parameters[0];
		_Rext = parameters[1];
		_Rint = (1 - parameters[2]) * parameters[1];
		
		double Inorm = 2*_V0/(M_PI * (_Rext*_Rext - _Rint*_Rint));
				
		// Allocate if necessary
		if (alpha.size()!=beta.size() || beta.size()!=I.size() || I.size()!=size_t(Ntot))
		{
			alpha.resize(Ntot);
			beta.resize(Ntot);
			I.resize(Ntot);
		}
		for (size_t j=0; j<alpha.size(); ++j)
		{
			if (alpha[j].size()!=beta[j].size() || beta[j].size()!=I[j].size() || I[j].size()!=size_t(Ntot))
			{
				alpha[j].resize(Ntot,0.0);
				beta[j].resize(Ntot,0.0);
				I[j].resize(Ntot,0.0);
			}
		}

		// Fill array with new image
		for (size_t j=0; j<alpha.size(); ++j)
		{
			for (size_t k=0; k<alpha[j].size(); ++k)
			{
				alpha[j][k] = ((double(j)-0.5*double(Ntot)+0.5)*_Rext*2.0*Rs/double(Ntot));
				beta[j][k]  = ((double(k)-0.5*double(Ntot)+0.5)*_Rext*2.0*Rs/double(Ntot));
				
				I[j][k] = 0.0;
				
				if (alpha[j][k]*alpha[j][k] + beta[j][k]*beta[j][k] <= _Rext*_Rext) {
					I[j][k] = 0.5 * (alpha[j][k]/_Rext + 1) ;
				}
				
				if ( ((alpha[j][k])*(alpha[j][k]) + (beta[j][k])*(beta[j][k])) <= _Rint*_Rint) {
					I[j][k] = 0.0;
				}
				
				I[j][k] *= Inorm;
				
			}
		}
	
	}
	
	
	double model_image_slashed_ring::BesselJ0(double x)
	{
		double ax,z;
		double xx,y,ans,ans1,ans2;
		
		if ((ax=fabs(x)) < 8.0) {
			y=x*x;
			ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7 
				+y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
			ans2=57568490411.0+y*(1029532985.0+y*(9494680.718 
				+y*(59272.64853+y*(267.8532712+y*1.0))));
			ans=ans1/ans2;
		}
		
		else {
			z=8.0/ax;
			y=z*z;
			xx=ax-0.785398164;
			ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
				+y*(-0.2073370639e-5+y*0.2093887211e-6))); 
			ans2 = -0.1562499995e-1+y*(0.1430488765e-3
				+y*(-0.6911147651e-5+y*(0.7621095161e-6
				-y*0.934945152e-7)));
			ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
		}
		return ans; 
	}

	
	
	double model_image_slashed_ring::BesselJ1(double x)
	{
		double ax,z;
		double xx,y,ans,ans1,ans2;

		if ((ax=std::fabs(x)) < 8.0) {
			y=x*x;
			ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
				+y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
			ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
				+y*(99447.43394+y*(376.9991397+y*1.0))));
			ans=ans1/ans2;
		} else {
			z=8.0/ax;
			y=z*z;
			xx=ax-2.356194491;
			ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
				+y*(0.2457520174e-5+y*(-0.240337019e-6))));
			ans2=0.04687499995+y*(-0.2002690873e-3
				+y*(0.8449199096e-5+y*(-0.88228987e-6
				+y*0.105787412e-6)));
			ans=std::sqrt(0.636619772/ax)*(std::cos(xx)*ans1-z*std::sin(xx)*ans2);
			if (x < 0.0) ans = -ans;
		}
		return ans;

	}


	double model_image_slashed_ring::BesselJ2(double x)
	{
		return (2*BesselJ1(x)/x - BesselJ0(x));
	}
	
	
	std::complex<double> model_image_slashed_ring::complex_visibility(double u, double v)
		{
			double ru = u*std::cos(_position_angle) + v*std::sin(_position_angle);
			double rv = -u*std::sin(_position_angle) + v*std::cos(_position_angle);
			
			ru *= -1.0; //Reflection to have the image look the way it's seen in the sky
				
			double k = 2.*M_PI*std::sqrt(ru*ru + rv*rv);// + 1.e-15*_Rext;
			double I0 = 2.*_V0/(M_PI * (_Rext*_Rext - _Rint*_Rint));
				
			const std::complex<double> i(0.0,1.0);
			std::complex<double> V;
			
			V = (M_PI*I0/k) * (_Rext*BesselJ1(k*_Rext) - _Rint*BesselJ1(k*_Rint))
			+ (i*M_PI*I0/(2.*k*k)) * 2.*M_PI*ru * ( _Rext*BesselJ0(k*_Rext) - _Rext*BesselJ2(k*_Rext) - 2.*BesselJ1(k*_Rext)/k )
			- (i*M_PI*I0/(2.*k*k)) * 2.*M_PI*ru * ( _Rint*BesselJ0(k*_Rint) - _Rint*BesselJ2(k*_Rint) - 2.*BesselJ1(k*_Rint)/k ) * (_Rint/_Rext);
					
			return ( V );
		}
		
	
	double model_image_slashed_ring::visibility_amplitude(datum_visibility_amplitude& d, double acc)
	{
		if (_use_analytical_visibilities)
		{
			return ( std::abs(complex_visibility(d.u, d.v)) );
		}
		
		else
		{
			return ( model_image::visibility_amplitude(d,acc) );
		}
		
	}
	
	double model_image_slashed_ring::closure_phase(datum_closure_phase& d, double acc)
	{
		if (_use_analytical_visibilities)
		{
			std::complex<double> V1 = complex_visibility(d.u1,d.v1);
			std::complex<double> V2 = complex_visibility(d.u2,d.v2);
			std::complex<double> V3 = complex_visibility(d.u3,d.v3);
			
			std::complex<double> B123 = V1*V2*V3;
						
			return ( std::imag(std::log(B123))*180.0/M_PI );
		}
		
		else
		{
			return ( model_image::closure_phase(d,acc) );
		}
	}


	double model_image_slashed_ring::closure_amplitude(datum_closure_amplitude& d, double acc)
	{
		if (_use_analytical_visibilities)
		{
			double V1 = std::abs(complex_visibility(d.u1,d.v1));
			double V2 = std::abs(complex_visibility(d.u2,d.v2));
			double V3 = std::abs(complex_visibility(d.u3,d.v3));
			double V4 = std::abs(complex_visibility(d.u4,d.v4));
			
			double V1234 = (V1*V3)/(V2*V4);
			
			return ( V1234 );
		}
		else
		{
			return ( model_image::closure_amplitude(d,acc) );
		}
	}
	
	
	std::complex<double> model_image_slashed_ring::visibility(datum_visibility& d, double acc)
	{
		if (_use_analytical_visibilities)
		{
			return ( complex_visibility(d.u,d.v) );
		}
		else
		{
			return ( model_image::visibility(d,acc) );
		}
	}


};
