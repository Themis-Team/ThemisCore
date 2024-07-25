/*
HYP: this is an example for hdf5 + mpi image
*/ 
//#include "../hdf5_setup/iharm_def.h"
#include "../hdf5_setup/iharm_load.h"
//#include "../hdf5_setup/iharm_func.h"
//===should always include the above headers

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <iostream>
using namespace std;
#include "mpi.h"
#define N	6
#define PI 3.141592654
#define C_G						  6.67259e-8	     //6.67259e-08
#define C_c						  2.99792458e10  //2.99792458e10  //2.99792458e+10
#define C_mSun					1.989e+33
#define C_rgeo          (C_G*C_mSun/C_c/C_c) //1.4774e+05        //C_G*C_msun/C_c/C_c
#define C_M87_mbh				(1.233e43/C_mSun)     //6.2e9             //mass of the black hole (M87)
#define C_M87_d				  16.9e6             //distance to  the black hole (M87), in unit of pc


static void GRRT(double *data, int x1, int y1);
static void geodesic(double *y, double *dydx);
static void initial(double *y0, double *ydot0, double x, double y);
static void rkstep(double *y, double *dydx, double h, double *yout, double *yerr);
static double rkqs(double *y, double *dydx, double htry, double escal, double *yscal, double *hdid);
static double inner_orbit(void);
static void datawrite_gnuplot(double *buf);
static void datawrite_aips(double *buf);
static void datawrite_header(void);
static void master(int numslaves);
static void slave(int numslaves, int rank);
static void gnuplot_script(void);
static double redshift(double r, double theta,double pr);
static double distribution_fun(double r);

static int count=0;
//static const double a; //requires an initializer
static double a;
static char outputfile_gnuplot[]="output.txt";
static const int size = 160;
static double inclination = 85.;
static double freq_obs=230e9;
static double r0, theta0;
static double a2;
static double Rhor, Rmstable; 
//static double Rdisk=29.09; //58.2;
//static double image_ratio=8.;  //image_size=image_ratio*Rdisk;
static double Rdisk=15.; //58.2;
static double image_ratio=1.5;  //image_size=image_ratio*Rdisk;
static double L, kappa;
static double normalization;

static char outputfile_aips[]="for_aips.txt";
static char outputfile_header[]="header.txt";
static double BHmass=6.6;  //e9 Msun
FILE *fp1;
FILE *fp2;
FILE *fp3;
double MKS_coeff[9]; //coefficient for MKS coordinate setup
int num_N[3];        //N1,N2,N3

int main(int argc, char** argv) 
{

   //cout << "You have entered " << argc
   //      << " arguments:" << "\n";
  
  //  for (int i = 0; i < argc; ++i)
  //      cout << argv[i] << "\n";

	int numprocs;
	int rank;
  
  char hdf5data_loc[100];
	fp1=fopen(argv[1],"r");
	  fscanf(fp1,"%s",hdf5data_loc);
    printf("===data location: %s\n",hdf5data_loc);
  fclose(fp1);
             
	/* in case the output is exist already*/
	fp1=fopen(outputfile_gnuplot,"w");
	fprintf(fp1,"# file for gnuplot\n");
        fclose(fp1);


	/* Initialize MPI */
	
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);  /* get number of processes */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);      /* get current process id */

  double spin=hdf5_data_load(MKS_coeff,num_N,hdf5data_loc);
  
  if (rank==0)
    printf("**********spin=%f\n",spin);
  a=spin;
  
   printf("===from h5 data: rho=%e \n",data[10][10][10][dRHO]);
   
	if (numprocs == 1) errx(1, "No! We need more than one rank!!");
	
	//else
	//	printf( "Hello world from process %d of %d\n", rank, numprocs );
	
	
	if (!rank)
	{
		master(numprocs - 1);
	}
	
	else
	{
		slave(numprocs - 1, rank - 1);
	}
       
       
	
	
	MPI_Finalize();
   
	
	return 0;
}



static void datawrite_gnuplot(double *buf)
{
// generate file for the use of gnuplot	
	int j,k;

	j=count;
	fp1=fopen(outputfile_gnuplot,"a");
	for(k=0;k< size; k++)
	{
	
	//if(buf[k]==0.)	printf("%d [%.0f,%.0f]\n",k,buf[k],buf[k+1]);
	fprintf(fp1,"%d %d %f\n",k ,j , buf[k]);
	}
	fclose(fp1);
	count++;
}

static void datawrite_aips(double *buf)
{
// generate file for the use of AIPS
       int i;
       fp2=fopen(outputfile_aips,"a");
       for (i=0;i<size;i++)
       {
	       fprintf(fp2,"%.5f ",buf[i]);
         normalization=normalization+buf[i];
       }
       fprintf(fp2,"\n");
       fclose(fp2);

}

static void datawrite_header(void)
{
	  fp1=fopen(outputfile_header,"w");
    fprintf(fp1,"! Required keywords\n"); 
    fprintf(fp1,"NAXIS = 4  DIM = %d,%d,1,1  FORMAT='%dF8.5'\n",size,size,size);
    fprintf(fp1,"! Optional keywords\n");
    fprintf(fp1,"OBJECT='M 87'  TELESCOP='GLT'  UNITS='JY/PIX'\n");
    fprintf(fp1,"INSTRUME='230GHz'\n");
    fprintf(fp1,"OBSERVER='GLT'\n");
    fprintf(fp1,"EPOCH=2000.0\n");
    fprintf(fp1,"SCALE=%.9f\n",1./normalization);
    fprintf(fp1,"CRTYPE = 'RA---SIN','DEC--SIN','FREQ    ','STOKES  '\n");
    fprintf(fp1,"CRVAL =  -1.722941835811D+02,  1.239117914251D+01, 230D+9, 1.000000000000D+00\n");
    

//    fprintf(fp1,"CRINC=-%.2e, %.2e, 1D7 ,1\n",2.5*Rdisk/(double)size*(BHmass/6.)*3.6*1.e-6/3600.0,2.5*Rdisk/(double)size*3.6e-6/3600.0);
    fprintf(fp1,"CRINC=-%.2e, %.2e, 1D7 ,1\n",image_ratio*Rdisk/(double)size*(BHmass/6.)*3.6*1.e-6/3600.0,image_ratio*Rdisk/(double)size*(BHmass/6.)*3.6*1.e-6/3600.0);
    fprintf(fp1,"! data size =%d  X  %d pixels  ==>  %.2e microarcsec/pixel \n",size,size,image_ratio*Rdisk/(double)size*(BHmass/6.)*3.6);
    fprintf(fp1,"! Model Informatino: BH Mass = %fe9   spin = %.3f    i = %f (degree) \n",BHmass,a,inclination);
    
    fprintf(fp1,"! Model Information: outer edge of the disk =%f (GM/c^{2})\n",Rdisk);
 //   fprintf(fp1,"! Model Information: total model image size =%f X %f (GM/c^{2} X  GM/c^{2})\n", 2.5*Rdisk, 2.5*Rdisk);
    fprintf(fp1,"! Model Information: total model image size =%f X %f (GM/c^{2} X  GM/c^{2})\n", image_ratio*Rdisk, image_ratio*Rdisk);
 
    fprintf(fp1,"CRROT =          0.0000,         0.0000,         0.0000,         0.0000\n");
    fprintf(fp1,"CRREF=       %f,          %f,         0.0000,         1.0000\n",(size+1)/2.0,(size+1)/2.0);
    fprintf(fp1,"/\n");

    fclose(fp1);
}

static void gnuplot_script(void)
{
       double end;
       
       //if (BHmass==3)
       //end=10./(2.5*Rdisk*1.8/size);
       
       //if (BHmass==6)
       //end=10./(2.5*Rdisk*3.6/size);
       
       //end=5./(2.5*Rdisk/size); // for 5 GM/c^2
       end=5./(image_ratio*Rdisk/size); 
       
       fp3=fopen("plot.gp","w");
       
       fprintf(fp3,"reset\n");
       fprintf(fp3,"set terminal postscript color enhanced\n");
       fprintf(fp3,"set output \"demo_image.eps\"\n");
       //if (BHmass==3)
       //fprintf(fp3,"set title \" Intensity Map  (%d x %d pixels):\\nM=3e9 M_{sun}  a=%.3f,  i = %.2f degree\\n\" \n",size,size,a,inclination);
       
       //if (BHmass==6)
       //fprintf(fp3,"set title \" Intensity Map  (%d x %d pixels):\\nM=6e9 M_{sun}  a=%.2f,  i = %.2f degree\\n\" \n",size,size,a,inclination);
       
       fprintf(fp3,"set title \" Intensity Map  (%d x %d pixels):\\n a=%f,  i = %f degree\\n\" \n",size,size,a,inclination);
       
       fprintf(fp3,"set xrange [0:%d]\n",size);
       fprintf(fp3,"set yrange [0:%d]\n",size);
       fprintf(fp3,"set palette rgb 21,22,23\n");
       //#set pm3d map
       //set palette rgb 34,35,36
       //#set pal negative
       //#set pal gray
       fprintf(fp3,"set view map\n");
       fprintf(fp3,"set size ratio 1\n");
       fprintf(fp3,"#unset colorbox\n");
       fprintf(fp3,"unset xtics\n");
       fprintf(fp3,"unset ytics\n");
       fprintf(fp3,"set style line 2 lt 3 lw 2\n");
       fprintf(fp3,"set style arrow 8 heads size screen 0.008,90 ls 2\n");
       fprintf(fp3,"set arrow from %f,%f to %f,%f as 8\n",size/4.,-size/10.,size/4.+end,-size/10.);
       fprintf(fp3,"set label \'5 GM/c^2\' at %f,%f right\n",size/5.,-size/10.);
       
       fprintf(fp3,"splot \"for_gnuplot.txt\" not  w dots palette\n");
     
       fclose(fp3);
       
}

/* The master process handles the IO */

static void master(int numslaves)
{

	int i,j,k;
	double x,y,z;
        int source;

	MPI_Status status;
        printf("===========start!===========\n");

   printf("master:a=%f\n",a);
   
           
        normalization=0.;
	double *buf = (double*) calloc(size, sizeof(double));
        if (!buf) errx(1, "out of memory\n");



	// Wait for size rows of data 
	for (i = 0; i < size; i++)
	{


	        source=(i%numslaves)+1;
	        
        	MPI_Recv(buf, size, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, &status);
                
		printf("complete %4d / %4d ",i+1,size);
	  //      printf(" ==>  i=%d  source=%d now, ",i, source);
	  //      printf( "done by slave %d (rank %d)!",status.MPI_SOURCE-1,status.MPI_SOURCE);
	        printf("\n");
		
   
   
		datawrite_gnuplot(buf);


	}

    
	printf("==========done!=========\n");

	free(buf);
	
}
 


/* Slave compute processes send their results to the master */

static void slave(int numslaves, int rank)
{
	MPI_Request rq = MPI_REQUEST_NULL;

	int i, j;
 



  double *buf = (double*) calloc(size,sizeof(double));	
	if (!buf) errx(1,"Out of memory\n");


printf("===slave %d: from h5 data: rho=%e \n",rank, data[10][10][10][dRHO]);
        

	r0 = 1000.0;
	theta0 = (PI/180.0) * inclination;

	a2 = a*a;

	Rhor = 1.0 + sqrt(1.0-a2) + 1e-5;
	//Rdisk = 20.0;
	Rmstable = inner_orbit();
	
        printf( "slave %d start working!\n",rank);

	
	for (j = rank; j < size; j += numslaves)
	{
		for (i = 0; i < size; i++)
		{
		      GRRT(&buf[i], i, j);
				     

		}
 
		// Send the new data (to Master)		
		MPI_Send(buf, size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

	 }
	

	free(buf);
}





static void GRRT(double *data, int x1, int y1)
{
	double htry = 0.5, escal = 1e11, hdid = 0.0, hnext = 0.0;

//	double range = 0.0025 * Rdisk / (size - 1.0);
 
  double range = 0.001 * image_ratio * Rdisk / (size - 1.0);


	double y[N], dydx[N], yscal[N], ylaststep[N];
 
 	double redshift_value, intensity_value;

    double ds=0.;
    double dI=0.;
    double dtau=0.;

 
 
	//int side;
	int i;
	int indirect=0;
	double r_surface,phi_surface;

	initial(y, dydx, (x1 - (size + 1.0) / 2.) * range, (y1 - (size + 1.0) / 2.) * range);


  double zzz;
	double stoke_j[4];
	double stoke_a[4];
	double faraday[4];
 
 
	while (1)
	{
		memcpy(ylaststep, y, N * sizeof(double));
        //fprintf(fp1,"%f  %f %f  %f %f\n", y[0], y[1], y[2], htry, hdid);
		geodesic(y, dydx);

		for (i = 0; i < N; i++)
		{
			yscal[i] = fabs(y[i]) + fabs(dydx[i] * htry) + 1.0e-3;
		}

		hnext = rkqs(y, dydx, htry, escal, yscal, &hdid);
   
   
   //***************** put accretion model here ********************/
   
  if ((ylaststep[1]-PI/2.)*(y[1]-PI/2.)<0.)
        {
        indirect=indirect+1;                                       
        }


           if( y[0]<50.)
        {
        double cor_KS[3];//r,th,phi
 //==loation in KS coordinate
 	//========================================	
 	cor_KS[0] = y[0];
  cor_KS[1] = acos(cos(cor_KS[1]));
  cor_KS[2] = fmod(atan2(sin(cor_KS[2]),cos(cor_KS[2]))+2.*PI,2.*PI);
  
  	//===interpolate the data at the given point: construct bil_value
	//========================================
	//get_Nloc(cor_KS, cor_N, num_N, cor_set, bil_value,data);

        // zzz=get_thersyn_coeff_HARM(stoke_j, stoke_a, faraday, Variables, VariablesIn, y, data,cor_set, num_N);

        zzz=1.;
        
        double freq_local=zzz*freq_obs;
	      double j_nu, alpha_nu;

	      //j_nu = stoke_j[0];
	      //alpha_nu = stoke_a[0];

             
		    ds   = htry;
		    dtau =  dtau   +  ds*C_M87_mbh*C_rgeo*alpha_nu*zzz;  
		    dI   =  dI	   +  ds*C_M87_mbh*C_rgeo*j_nu/freq_local/freq_local/freq_local*exp(-dtau)*zzz;  
				
        //=== test: free-fall
        /*
        redshift_value=redshift(y[0],y[1],y[4]);
        ds=htry;
        dI=dI+ds*distribution_fun(y[0])/redshift_value/redshift_value;
        */
        
			}
       

        
      
	
    

//Inside the hole, or escaped to infinity 

		if ((y[0] < Rhor) || (y[0] > r0))
		{
          
			data[0]=dI;


			return;
		}
		
		
		htry = hnext;
	}
}


static void geodesic(double *y, double *dydx)
{
	double r, theta, pr, ptheta;

	r = y[0];
	theta = y[1];
	pr = y[4];
	ptheta = y[5];

	double r2 = r*r;
	double twor = 2.0*r;

	double sintheta, costheta;
	sintheta=sin(theta);
	costheta=cos(theta);
	//sincos(theta, &sintheta, &costheta);
	double cos2 = costheta*costheta;
	double sin2 = sintheta*sintheta;

	double sigma = r2+a2*cos2;
	double delta = r2-twor+a2;
	double sd = sigma*delta;
	double siginv = 1.0/sigma;
	double bot = 1.0/sd;

	/* Prevent problems with the axis */
	if (sintheta < 1e-8)
	{
		sintheta = 1e-8;
		sin2 = 1e-16;
	}

	dydx[0] = -pr*delta*siginv;
	dydx[1] = -ptheta*siginv;
	dydx[2] = -(twor*a+(sigma-twor)*L/sin2)*bot;
	dydx[3] = -(1.0+(twor*(r2+a2)-twor*a*L)*bot);
	dydx[4] = -(((r-1.0)*(-kappa)+twor*(r2+a2)-2.0*a*L)*bot-2.0*pr*pr*(r-1.0)*siginv);
	dydx[5] = -sintheta*costheta*(L*L/(sin2*sin2)-a2)*siginv;
}

/* Initial Conditions for Ray */
static void initial(double *y0, double *ydot0, double x, double y)
{

  x=-x;
	y0[0] = r0;
	y0[1] = theta0;
	y0[2] = 0;
	y0[3] = 0;
	y0[4] = cos(y)*cos(x);
	y0[5] = sin(y)/r0;

	double sintheta, costheta;
	sintheta=sin(theta0);
	costheta=cos(theta0);
	//sincos(theta0, &sintheta, &costheta);
	double cos2 = costheta*costheta;
	double sin2 = sintheta*sintheta;

	double rdot0 = y0[4];
	double thetadot0 = y0[5];

	double r2 = r0 * r0;
	double sigma = r2 + a2*cos2;
	double delta = r2 - 2.0 * r0 + a2;
	double s1 = sigma - 2.0 * r0;

	y0[4]= rdot0*sigma/delta;
	y0[5]= thetadot0*sigma;

	ydot0[0] = rdot0;
	ydot0[1] = thetadot0;
	ydot0[2] = cos(y)*sin(x)/(r0*sin(theta0));

	double phidot0 = ydot0[2];
	double energy2 = s1*(rdot0*rdot0/delta+thetadot0*thetadot0)
					+ delta*sin2*phidot0*phidot0;

	double energy = sqrt(energy2);

	/* Rescale */
	y0[4] = y0[4]/energy;
	y0[5] = y0[5]/energy;

	/* Angular Momentum with E = 1 */
	L = ((sigma*delta*phidot0-2.0*a*r0*energy)*sin2/s1)/energy;

	kappa = y0[5]*y0[5]+a2*sin2+L*L/sin2;

	/* Hack - make sure everything is normalized correctly by a call to geodesic */
	geodesic(y0, ydot0);
}

static void rkstep(double *y, double *dydx, double h, double *yout, double *yerr)
{
	int i;

	double ak[N];

	double ytemp1[N], ytemp2[N], ytemp3[N], ytemp4[N], ytemp5[N];

	for (i = 0; i < N; i++)
	{
		double hdydx = h * dydx[i];
		double yi = y[i];
		ytemp1[i] = yi + 0.2 * hdydx;
		ytemp2[i] = yi + (3.0/40.0) * hdydx;
		ytemp3[i] = yi + 0.3 * hdydx;
		ytemp4[i] = yi -(11.0/54.0) * hdydx;
		ytemp5[i] = yi + (1631.0/55296.0) * hdydx;
		yout[i] = yi + (37.0/378.0) * hdydx;
		yerr[i] = ((37.0/378.0)-(2825.0/27648.0)) * hdydx;
	}

	geodesic(ytemp1, ak);

	for (i = 0; i < N; i++)
	{
		double yt = h * ak[i];
		ytemp2[i] += (9.0/40.0) * yt;
		ytemp3[i] -= 0.9 * yt;
		ytemp4[i] += 2.5 * yt;
		ytemp5[i] += (175.0/512.0) * yt;
	}

	geodesic(ytemp2, ak);

	for (i = 0; i < N; i++)
	{
		double yt = h * ak[i];
		ytemp3[i] += 1.2 * yt;
		ytemp4[i] -= (70.0/27.0) * yt;
		ytemp5[i] += (575.0/13824.0) * yt;
		yout[i] += (250.0/621.0) * yt;
		yerr[i] += ((250.0/621.0)-(18575.0/48384.0)) * yt;
	}

	geodesic(ytemp3, ak);

	for (i = 0; i < N; i++)
	{
		double yt = h * ak[i];
		ytemp4[i] += (35.0/27.0) * yt;
		ytemp5[i] += (44275.0/110592.0) * yt;
		yout[i] += (125.0/594.0) * yt;
		yerr[i] += ((125.0/594.0)-(13525.0/55296.0)) * yt;
	}

	geodesic(ytemp4, ak);

	for (i = 0; i < N; i++)
	{
		double yt = h * ak[i];
		ytemp5[i] += (253.0/4096.0) * yt;
		yerr[i] -= (277.0/14336.0) * yt;
	}

	geodesic(ytemp5, ak);

	for (i = 0; i < N; i++)
	{
		double yt = h * ak[i];
		yout[i] += (512.0/1771.0) * yt;
		yerr[i] += ((512.0/1771.0)-0.25) * yt;
	}
}

static double rkqs(double *y, double *dydx, double htry, double escal, double *yscal, double *hdid)
{
	int i;

	double hnext;

	double errmax, h = htry, htemp;
	double yerr[N], ytemp[N];

	while (1)
	{
		rkstep(y, dydx, h, ytemp, yerr);

		errmax = 0.0;
		for (i = 0; i < N; i++)
		{
			double temp = fabs(yerr[i]/yscal[i]);
			if (temp > errmax) errmax = temp;
		}

		errmax *= escal;
		if (errmax <= 1.0) break;

		htemp = 0.9 * h / sqrt(sqrt(errmax));

		h *= 0.1;

		if (h >= 0.0)
		{
			if (htemp > h) h = htemp;
		}
		else
		{
			if (htemp < h) h = htemp;
		}
	}

	if (errmax > 1.89e-4)
	{
		hnext = 0.9 * h * pow(errmax, -0.2);
	}
	else
	{
		hnext = 5.0 * h;
	}

	*hdid = h;

	memcpy(y, ytemp, N * sizeof(double));

	return hnext;
}




static double inner_orbit(void)
{
	double z1 = 1+cbrt(1-a2)*(cbrt(1+a)+cbrt(1-a));
	double z2 = sqrt(3*a2+z1*z1);
	return 3+z2-sqrt((3-z1)*(3+z1+2*z2));
}

static double redshift(double r,double theta,double pr)
{
       double ut,uphi,ur,zeta,n,d;
      // double delta=r*r-2.*r+a*a;
      // double sigma=r*r+a*a*cos(theta)*cos(theta);
      
       
       
       	double r2 = r*r;
	double twor = 2.0*r;
       	double sintheta, costheta;
	sintheta=sin(theta);
	costheta=cos(theta);
	//sincos(theta, &sintheta, &costheta);
	double cos2 = costheta*costheta;
	double sin2 = sintheta*sintheta;



	/* Prevent problems with the axis */
//	if (sintheta < 1e-8)
//	{
//		sintheta = 1e-8;
//		sin2 = 1e-16;
//	}
	
		double sigma = r2+a2*cos2;
	double delta = r2-twor+a2;
	 double bigA=(r*r+a*a)*(r*r+a*a)-a*a*delta*sin2;
	 
       ut=bigA/delta/sigma;
       uphi=2.*a*r/delta/sigma;
       ur=-sqrt((r*r+a*a)*r*2.)/sigma;
       n=-ut+L*uphi+pr*ur;
       d=-1.;    
       return n/d;       
}

static double distribution_fun(double r)
{
      //double
      //return 1./r/r;  
      
      //case 1
      return exp(-r*r/(0.5*Rdisk));
      
      //case2
      //return exp(-r*r/(0.8*Rdisk));       
       
}

