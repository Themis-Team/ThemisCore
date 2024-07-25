//#include "Odyssey_def.h"
#include "iharm_def.h"
#include "BL_metric.h"
//#ifndef ODYSSEY_IHARM_FUNC_H
//#define ODYSSEY_IHARM_FUNC_H

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
//
// functions for processing harm data																								  
//
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 

//============
//FUNTION: lower the index [BL] and get convariant 4-vector
//============
void lower_bl(double *y, double *ucon, double *ucov)
{
ucov[0] = g00(y)*ucon[0] + g03(y)*ucon[3];
ucov[1] = g11(y)*ucon[1]; 
ucov[2] = g22(y)*ucon[2]; 
ucov[3] = g33(y)*ucon[3] + g03(y)*ucon[0] ;
}


//============
//FUNTION: obtain contrvariant 4-velocity 4-magnetic field, and return redshift [BL]
//============
double  get_u_B_bl(double *y, int *cor_N, int *num_N, double *cor_set, double *bil_val, double *u_BL, double *b_BL){


double ui=bil_val[dU1];
double uj=bil_val[dU2];
double uk=bil_val[dU3];

double bi=bil_val[dB1];
double bj=bil_val[dB2];
double bk=bil_val[dB3];


double r=y[0];
double theta=y[1];

//get x2 coordinate
double startx2    = cor_set[2];
double dx2        = cor_set[3];
double hslope     = cor_set[6];


double n2_below=(double)cor_N[1];
double n2_above=(double)cor_N[1]+1;
double th,x2_mid;
double x2_low,x2_high;

//x2_low = startx2 + (n2_below + 0.) * dx2;
//th = M_PI*x2_low+(1. - hslope)/2.* sin(2.* M_PI* x2_low);
//x2_high = startx2 + (n2_above + 0.) * dx2;
//th = M_PI*x2_high+(1. - hslope)/2.* sin(2.* M_PI* x2_high);
th=0.5;
x2_low=0.;
x2_high=startx2+((double)NN2+cen_off)*dx2;
	
//==this accuracy should be enough for SIZE<3000
//==a higher accuracy is needed for a  higher resolution, otherwise q2 would become -nan
double accuracy=1e-10;
for (int i=0;fabs(th-theta)>accuracy;i++){
x2_mid=(x2_low+x2_high)/2.;
th = M_PI*x2_mid+(1. - hslope)/2.* sin(2.* M_PI* x2_mid);
if (th>theta)
    x2_high=x2_mid;
else
    x2_low=x2_mid;
	
if (i==100)
   printf("cannot find x2_mid after 100 trials!\n");
if (i==1000)
   printf("cannot find x2_mid after 1000 trails! giving up now!\n");
}



//KS metric: ignore t terms 
double sigma=r*r+a2*cos(theta)*cos(theta);
double delta=r*r-2.*r+a2;
double beta_r=2.*r/sigma;
double AA=r*r+a2*(1.+2.*r*sin(theta)*sin(theta)/sigma);
double grr_ks=1.+beta_r;
double gthth_ks=sigma;
double gphph_ks= AA*sin(theta)*sin(theta);
double grph_ks=-a*(1.+beta_r)*sin(theta)*sin(theta);
double gtt_ks=-(1.-beta_r);
double gtr_ks=beta_r;
double gtph_ks=-a*beta_r*sin(theta)*sin(theta);

//MKS metric: ignore t terms 
double KS_MKS_r = r ;
 
double grr_mks=grr_ks*KS_MKS_r*KS_MKS_r;
double gphph_mks= gphph_ks;
double grph_mks=grph_ks*KS_MKS_r;
double gtt_mks=gtt_ks;
double gtr_mks=gtr_ks*KS_MKS_r;
double gtph_mks=gtph_ks;


//MKS metric: terms related to x2
double KS_MKS_th = M_PI * (1.+(1. - hslope)*cos(2.*M_PI*x2_mid)) ;	
double gthth_mks=gthth_ks*KS_MKS_th*KS_MKS_th;	
double q2=grr_mks*ui*ui+gthth_mks*uj*uj+gphph_mks*uk*uk+2.*grph_mks*ui*uk;
	
if(isnan(q2)){
	
printf("get_u_B_bl: q2=nan ==> increase accuracy to find x2, ");
accuracy=1e-15;
for (int i=0;fabs(th-theta)>accuracy;i++){
x2_mid=(x2_low+x2_high)/2.;
th = M_PI*x2_mid+(1. - hslope)/2.* sin(2.* M_PI* x2_mid);
if (th>theta)
    x2_high=x2_mid;
else
    x2_low=x2_mid;
	
if (i==100)
   printf("cannot find x2_mid after 100 trials!\n");
if (i==1000)
   printf("cannot find x2_mid after 1000 trails! giving up now!\n");
}

KS_MKS_th = M_PI * (1.+(1. - hslope)*cos(2.*M_PI*x2_mid)) ;	
gthth_mks=gthth_ks*KS_MKS_th*KS_MKS_th;	
q2=grr_mks*ui*ui+gthth_mks*uj*uj+gphph_mks*uk*uk+2.*grph_mks*ui*uk;		
}
	
//construct 4-vel [MKS]	
double gamma=sqrt(1.+q2);
double v2=q2/gamma/gamma;
if (v2>1)
printf("return zzz: v2>1 when r=%f th=%f\n",r,theta);


double alpha2_ana=1./(1.+beta_r);
double alpha_ana=sqrt(alpha2_ana);
double beta1_ana=beta_r/(1.+beta_r);
double beta2_ana=0.;
double beta3_ana=0.;

double ut_mks  = gamma/alpha_ana;
double ur_mks  = ui-ut_mks*beta1_ana/KS_MKS_r;
double uth_mks = uj-ut_mks*beta2_ana;
double uph_mks = uk-ut_mks*beta3_ana;


//construct 4-magnetic [MKS]
 double bt_mks=grr_mks*bi*ur_mks
               +gthth_mks*bj*uth_mks
               +gphph_mks*bk*uph_mks
               +grph_mks*bk*ur_mks
               +grph_mks*bi*uph_mks
               +gtr_mks*bi*ut_mks
               +gtph_mks*bk*ut_mks;
 
double br_mks=(bi+ur_mks*bt_mks)/ut_mks;
double bth_mks=(bj+uth_mks*bt_mks)/ut_mks;
double bph_mks=(bk+uph_mks*bt_mks)/ut_mks;



//construct 4-vel [MKS -> KS] 
 double ut_ks  = ut_mks;
 double ur_ks  = ur_mks*KS_MKS_r;
 double uth_ks = uth_mks*KS_MKS_th;
 double uph_ks = uph_mks;


double check_u2_ks=gtt_mks*ut_ks*ut_ks+
                2.*gtr_ks*ut_ks*ur_ks+
		2*gtph_ks*ut_ks*uph_ks+
		grr_ks*ur_ks*ur_ks+
		2*grph_ks*ur_ks*uph_ks+
		gthth_ks*uth_ks*uth_ks+
		gphph_ks*uph_ks*uph_ks;
   
//construct 4-vel [KS -> BL]
double coff1=2.*r/delta;
double coff2=a/delta;

u_BL[0]  =ut_ks-coff1*ur_ks; //u^t
u_BL[1]  =ur_ks;             //u^r
u_BL[2]  =uth_ks;
u_BL[3]  =uph_ks-coff2*ur_ks;



//construct 4-B [MKS -> KS] 
 double bt_ks  = bt_mks;
 double br_ks  = br_mks*KS_MKS_r;
 double bth_ks = bth_mks*KS_MKS_th;
 double bph_ks = bph_mks;
 

   
//construct 4-B [KS -> BL]

b_BL[0]  =bt_ks-coff1*br_ks; //b^t
b_BL[1]  =br_ks;             //b^r
b_BL[2]  =bth_ks;
b_BL[3]  =bph_ks-coff2*br_ks;


//computed redshift
        double E_inf=-1.;
       	double E_local=-u_BL[0]+y[4]*u_BL[1] + y[5]*u_BL[2]+L*u_BL[3];   
        
        double check_u2=g00(y)*u_BL[0]*u_BL[0]+
                        g11(y)*u_BL[1]*u_BL[1]+
                        g22(y)*u_BL[2]*u_BL[2]+
                        g33(y)*u_BL[3]*u_BL[3]+
                        2.*g03(y)*u_BL[0]*u_BL[3];

if(fabs(check_u2+1)>0.1 || fabs(check_u2_ks+1.)>0.1){
	printf("get_u_B_bl: WARNING!! zzz=%f, n2=%d n3=%d r=%f (rHor=%f) u2_bl=%f  u2_ks=%f\n",E_local/E_inf,cor_N[1],cor_N[2],r,1.+sqrt(1.-a2),check_u2,check_u2_ks);
}

if(isnan(E_local)){
       printf("get_u_B_bl: ERROR!! E_local=%f at (r,th,ph)=(%f,%f,%f), with dRHO=%e q2=%f alpha2=%f uijk=(%f, %f, %f) uBL=(%f, %f, %f, %f) bBL=(%f, %f, %f, %f)\n",E_local,y[0],y[1],y[2],bil_val[dRHO],q2,alpha2_ana,ui,uj,uk,u_BL[0],u_BL[1],u_BL[2],u_BL[3],b_BL[0],b_BL[1],b_BL[2],b_BL[3]);   
}
	     return E_local/E_inf;
             
}

//============
//FUNTION: obtain contrvariant 4-velocity 4-magnetic field, and return redshift [BL]
//============
 double get_pitch_b(double *y,  double *ucon_BL, double *bcon_BL, double *bcov_BL){

double kcov[4];
kcov[0]=-1.;   //p_t=E;
kcov[1]=y[4];
kcov[2]=y[5];
kcov[3]=L;


//==avoid numerical error close to horizon
double B2=fabs(dot(bcon_BL,bcov_BL));
//B=sqrt(B);
if (B2==0.){
   printf("...B=0....\n");   	
  return M_PI/2.;}
  
double K=-dot(kcov,ucon_BL);
//====??
//K=fabs(K); 


	
	
//double mu2=dot(kcov,bcon_BL)*dot(kcov,bcon_BL)/(K*K*B2+1e-15); // always >0 therefore artificially always 90>thetab>0
//double mu=sqrt(mu2);	

double mu=dot(kcov,bcon_BL)/(K*sqrt(B2)+1e-15);

//double approx=0.99;        
    if (fabs(mu) > 1.){
	    mu /= fabs(mu);
      //   mu=approx;
      //  if (mu<0.)
      //     mu=-mu;
         }
    if (isnan(mu))
	    printf("...isnan get_pitch_b...\n");   	
//return product;
return acos(mu);

}

/*
//============
//FUNTION: input coor, return KS
//============
 void get_KScoor(double* Variables, double* VariablesIn, double *cor_KS, int *cor_N, double *cor_set){

double startx1    = cor_set[0];
double dx1        = cor_set[1];
double startx2    = cor_set[2];
double dx2        = cor_set[3];
double startx3    = cor_set[4];
double dx3        = cor_set[5];
double hslope     = cor_set[6];

double n1= (double)(cor_N[0]);
double n2= (double)(cor_N[1]);
double n3= (double)(cor_N[2]);

//===compute BL coordinates
double x1,x2,x3;

//search for n1
x1 = startx1 + (n1 + cen_off) * dx1;
x2 = startx2 + (n2 + cen_off) * dx2;
x3 = startx3 + (n3 + cen_off) * dx3;

double r_KS =exp(x1);
double th_KS=M_PI*x2+(1. - hslope)/2.* sin(2.* M_PI* x2);
double ph_KS=x3;
	
cor_KS[0]=r_KS; 
cor_KS[1]=th_KS; 
cor_KS[2]=ph_KS;
}

*/
//============
//FUNTION: input KS, return coor and bilinear interpolation
//============
void  get_Nloc(double *cor_KS, int *cor_N, int *num_N, double *cor_set, double *bil_value, double* data){
double startx1    = cor_set[0];
double dx1        = cor_set[1];
double startx2    = cor_set[2];
double dx2        = cor_set[3];
double startx3    = cor_set[4];
double dx3        = cor_set[5];
double hslope     = cor_set[6];

int n1,n2, n3;

//===compute BL coordinates: r
double x1,x2,x3;
double r1,th1,ph1;
double r2,th2,ph2;

double dr,dth,dph; //>0
double Dr, Dth,Dph;

for (n1=0;n1<=(num_N[0]-1);n1++){
  x1 = startx1 + ((double)n1 + cen_off) * dx1;
  r1=exp(x1);
  x1 = startx1 + ((double)(n1+1.) + cen_off) * dx1;
  r2=exp(x1);
  if (cor_KS[0]>=r1 && r2>cor_KS[0]){
    Dr=r2-r1;
    dr=r2-cor_KS[0];
    break;
  }
}

//===compute BL coordinates: th

for (n2=0;n2<=NN2;n2++){

 	x2 = startx2 + ((double)(n2) + cen_off) * dx2;
	th2=M_PI*x2+(1. - hslope)/2.* sin(2.* M_PI* x2);

	dth=th2-cor_KS[1];
	    //intf("n2=%d  th2=%f th=%f\n",n2,th2, cor_KS[1]);
	if (dth>=0){
	        //printf("n2=%d th2=%f cor_KS[1]=%f  dth=%f > 0\n",n2,th2,cor_KS[1], dth);
		break;
	   }
	    
	}



//===compute BL coordinates: ph

for (n3=0;n3<=NN3;n3++){
	x3 = startx3 + ((double)n3 + cen_off) * dx3;
	ph2=x3;

	dph=ph2-cor_KS[2];

	if (dph>=0.){
	    	//printf("n3=%d ph2=%f cor_KS[2]=%f  dph=%f > 0\n",n3,ph2,cor_KS[2], dph);
		break;
	            }
			
	 }


//==== assing coordinates
cor_N[0]=n1; //for r
cor_N[1]=n2; //for th
cor_N[2]=n3; //for ph

int n1_far=n1+1;
int n1_near=n1;
int n2_far=n2%NN2;
int n2_near=(n2-1+NN2)%NN2;
int n3_far=n3%NN3;
int n3_near=(n3-1+NN3)%NN3;

//if (n2==0 || n2==NN2 || n3==0 || n3==NN3)
//printf("n2_far=%d n2_near=%d n3_near=%d n3_near=%d\n",n2_far,n2_near,n3_far,n3_near);

x2 = startx2 + ((double)(n2_near) + cen_off) * dx2;
th1=M_PI*x2+(1. - hslope)/2.* sin(2.* M_PI* x2);
Dth=th2-fmod(th1+M_PI,M_PI);

x3 = startx3 + ((double)(n3_near) + cen_off) * dx3;
ph1=x3;
Dph=ph2-ph1;
Dph=fmod(Dph+2.*M_PI,2.*M_PI);


double phA[N_readto],phB[N_readto],phC[N_readto],phD[N_readto];

/*
 * (1)linear interpolation in phi direction
 * [n3_near]        [n3_far]
 *  |           dph    |
 *  |       <--------->|
 *  |<------+--------->|
 *  |          Dph     |
 *  ph1     ph         ph2
 *   
 */

//pLOOP phA[j]=(data[n1_near][n2_near][n3_far][j]*(Dph-dph)+data[n1_near][n2_near][n3_near][j]*dph)/(Dph);
//pLOOP phB[j]=(data[n1_far][n2_near][n3_far][j]*(Dph-dph)+data[n1_far][n2_near][n3_near][j]*dph)/(Dph);
//pLOOP phC[j]=(data[n1_near][n2_far][n3_far][j]*(Dph-dph)+data[n1_near][n2_far][n3_near][j]*dph)/(Dph);
//pLOOP phD[j]=(data[n1_far][n2_far][n3_far][j]*(Dph-dph)+data[n1_far][n2_far][n3_near][j]*dph)/(Dph);

pLOOP phA[j]=(data[data_order(n1_near,n2_near,(n3_far),j)]*(Dph-dph)+data[data_order(n1_near,n2_near,n3_near,j)]*dph)/(Dph);
pLOOP phB[j]=(data[data_order((n1_far),n2_near,(n3_far),j)]*(Dph-dph)+data[data_order((n1_far),n2_near,n3_near,j)]*dph)/(Dph);
pLOOP phC[j]=(data[data_order(n1_near,(n2_far),(n3_far),j)]*(Dph-dph)+data[data_order(n1_near,(n2_far),n3_near,j)]*dph)/(Dph);
pLOOP phD[j]=(data[data_order((n1_far),(n2_far),(n3_far),j)]*(Dph-dph)+data[data_order((n1_far),(n2_far),n3_near,j)]*dph)/(Dph);


/*
 * (2)then use bilinear interpliation in (r,th) plan:
 * A:[n1_near][]    B:[n1_far][n2_near]
 *  |------------------|
 *  |  dD |    dC      |
 *  |-----+<----dr---->| Dth
 *  |    (r,th)        |
 *  | dB  |    dA      |
 *  |     dth          |
 *  |<---------------->|
 *          Dr
 * C:[][]           D:[][n2_far]
 *                      */

 double dA=dr*dth;
 double dB=(Dr-dr)*dth;
 double dC=dr*(Dth-dth);
 double dD=(Dr-dr)*(Dth-dth);

 pLOOP bil_value[j]=(phD[j]*dA+phC[j]*dB+phB[j]*dC+phA[j]*dD)/((r2-r1)*(th2-th1));
 //pLOOP bil_value[j]=data[n1][n2][n3][j];

}




//============
//FUNTION: obtain contrvariant 4-velocity and normalized 4-magnetic field [BL]
//         for Odyssey_pol_ray_tracing.h
//============
 void  get_u_nB_bl(double *y, int *cor_N, int *num_N, double *cor_set, double *bil_val, double *u_BL, double *b_BL){


double ui=bil_val[dU1];
double uj=bil_val[dU2];
double uk=bil_val[dU3];

double bi=bil_val[dB1];
double bj=bil_val[dB2];
double bk=bil_val[dB3];


double r=y[0];
double theta=y[1];

theta=acos(cos(theta));
//get x2 coordinate
double startx2    = cor_set[2];
double dx2        = cor_set[3];
double hslope     = cor_set[6];


//double n2_below=(double)cor_N[1];
//double n2_above=(double)cor_N[1]+1;
double th,x2_mid;
double x2_low,x2_high;
/*
x2_low = startx2 + (n2_below + 0.) * dx2;
th = M_PI*x2_low+(1. - hslope)/2.* sin(2.* M_PI* x2_low);
x2_high = startx2 + (n2_above + 0.) * dx2;
th = M_PI*x2_high+(1. - hslope)/2.* sin(2.* M_PI* x2_high);
*/

x2_low=0.;
x2_high=(double)NN2;
//==this accuracy should be enough for SIZE<3000
//==a higher accuracy is needed for a  higher resolution, otherwise q2 would become -nan
double accuracy=1e-12;
th=0.5;
for (int i=0;fabs(th-theta)>accuracy;i++){
x2_mid=(x2_low+x2_high)/2.;
th = M_PI*x2_mid+(1. - hslope)/2.* sin(2.* M_PI* x2_mid);
if (th>theta)
    x2_high=x2_mid;
else
    x2_low=x2_mid;

if (i==100)
   printf("cannot find x2_mid!\n");
}


//printf("inter th:%f   data_th=%f\n", th,theta);
//KS metric: ignore t terms 
double sigma=r*r+a2*cos(theta)*cos(theta);
double delta=r*r-2.*r+a2;
double beta_r=2.*r/sigma;
double AA=r*r+a2*(1.+2.*r*sin(theta)*sin(theta)/sigma);
double grr_ks=1.+beta_r;
double gthth_ks=sigma;
double gphph_ks= AA*sin(theta)*sin(theta);
double grph_ks=-a*(1.+beta_r)*sin(theta)*sin(theta);
double gtt_ks=-(1-beta_r);
double gtr_ks=beta_r;
double gtph_ks=-a*beta_r*sin(theta)*sin(theta);

//MKS metric: ignore t terms 
double KS_MKS_r = r ;
double KS_MKS_th = M_PI * (1.+(1. - hslope)*cos(2.*M_PI*x2_mid)) ;
 
double grr_mks  = grr_ks*KS_MKS_r*KS_MKS_r;
double gthth_mks= gthth_ks*KS_MKS_th*KS_MKS_th;
double gphph_mks= gphph_ks;
double grph_mks = grph_ks*KS_MKS_r;
double gtt_mks  = gtt_ks;
double gtr_mks  = gtr_ks*KS_MKS_r;
double gtph_mks = gtph_ks;



//construct 4-vel [MKS]
double q2=grr_mks*ui*ui+gthth_mks*uj*uj+gphph_mks*uk*uk+2.*grph_mks*ui*uk;
double gamma=sqrt(1.+q2);
double v2=q2/gamma/gamma;
if (v2>1)
printf("no return zzz: v2>1 when r=%f th=%f\n",r,theta);


double alpha2_ana=1./(1.+beta_r);
double alpha_ana=sqrt(alpha2_ana);
double beta1_ana=beta_r/(1.+beta_r);
double beta2_ana=0.;
double beta3_ana=0.;

double ut_mks  = gamma/alpha_ana;
double ur_mks  = ui-ut_mks*beta1_ana/KS_MKS_r;
double uth_mks = uj-ut_mks*beta2_ana;
double uph_mks = uk-ut_mks*beta3_ana;


//construct 4-magnetic [MKS]
 double bt_mks=grr_mks*bi*ur_mks
               +gthth_mks*bj*uth_mks
               +gphph_mks*bk*uph_mks
               +grph_mks*bk*ur_mks
               +grph_mks*bi*uph_mks
               +gtr_mks*bi*ut_mks
               +gtph_mks*bk*ut_mks;
 
double br_mks=(bi+ur_mks*bt_mks)/ut_mks;
double bth_mks=(bj+uth_mks*bt_mks)/ut_mks;
double bph_mks=(bk+uph_mks*bt_mks)/ut_mks;



//construct 4-vel [MKS -> KS] 
 double ut_ks  = ut_mks;
 double ur_ks  = ur_mks*KS_MKS_r;
 double uth_ks = uth_mks*KS_MKS_th;
 double uph_ks = uph_mks;


   

//construct 4-vel [KS -> BL]
double coff1=2.*r/delta;
double coff2=a/delta;

u_BL[0]  =ut_ks-coff1*ur_ks; //u^t
u_BL[1]  =ur_ks; //u^r
u_BL[2]  =uth_ks;
u_BL[3]  =uph_ks-coff2*ur_ks;



//construct 4-B [MKS -> KS] 
 double bt_ks  = bt_mks;
 double br_ks  = br_mks*KS_MKS_r;
 double bth_ks = bth_mks*KS_MKS_th;
 double bph_ks = bph_mks;
 

   
//construct 4-B [KS -> BL]

b_BL[0]  =bt_ks-coff1*br_ks; //b^t
b_BL[1]  =br_ks; //b^r
b_BL[2]  =bth_ks;
b_BL[3]  =bph_ks-coff2*br_ks;


double bcov_BL[4];
	
lower_bl(y, b_BL, bcov_BL);
double b2= dot(b_BL,bcov_BL);  
	if (b2<=0){
            b2=1e-6;
            //theta_b=0.;
            //nth=0.;
            } 
double b=sqrt(b2);;    

//normalize
b_BL[0]  =b_BL[0]/b; //b^t
b_BL[1]  =b_BL[1]/b; //b^r
b_BL[2]  =b_BL[2]/b;
b_BL[3]  =b_BL[3]/b;	
	
	
}


//#endif
