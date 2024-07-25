/******************************************************************************
 *                                                                            *
 * iharm_func.C                                                               *
 *                                                                            *
 * FUNCTIONS FOR processing iharm output(for MKS coordinate)                  *
 *                                                                            *
 ******************************************************************************/
#include "./hdf5_utils_func.h"
#include "iharm_def.h"
#include <hdf5.h>



//==================
//=== set up global parameters (if necessary)
//double L;
//============

double data[NN1][NN2][NN3][N_readto];

//============
//FUNTION: read in hdf5 data
//============
double hdf5_data_load(double *cor_set,int *num_N, char *data_dat){

double gam,t,tf;
int n_prim;
char metric[20];

int N1;
int N2;
int N3;

printf("===start reading hdf5 data: %s\n",data_dat);
hid_t HDF5_STR_TYPE = hdf5_make_str_type(20);
//hdf5_open(harmfname);
hdf5_open(data_dat);
hdf5_set_directory("/");

hdf5_read_single_val(&gam, "/header/gam", H5T_IEEE_F64LE);  // pointer to memory; H5T_IEEE_F64LE  -> little endian double
hdf5_read_single_val(&N1, "/header/n1", H5T_STD_I32LE);  // pointer to memory; H5T_IEEE_F64LE  -> little endian integer
hdf5_read_single_val(&N2, "/header/n2", H5T_STD_I32LE);  
hdf5_read_single_val(&N3, "/header/n3", H5T_STD_I32LE);
hdf5_read_single_val(&t, "/t", H5T_IEEE_F64LE);  
hdf5_read_single_val(&tf, "/header/tf", H5T_IEEE_F64LE);   
hdf5_read_single_val(&n_prim, "/header/n_prim", H5T_STD_I32LE); 
hdf5_read_single_val(&metric, "/header/metric", HDF5_STR_TYPE); 

printf("number of prim:%d\n",n_prim);
//printf("var_name=%s\n",var_name[0]);
printf("N1, N2, N3= %d %d %d\n",N1,N2,N3);

 if (N1!=NN1 || N2!=NN2 || N3!=NN3){
  printf("NN1, NN2, NN3= %d %d %d\n",NN1,NN2,NN3);
  printf("load data: ERROR!!! ===== redefine NNi (should =Ni) ====\n");
  //break;
}
printf("t/tf= %f/%f\n",t,tf);
printf("metric= %s \n",metric);

double Reh,Rin,Rout,a,hslope,mks_smooth,poly_alpha,poly_xt,startx1,startx2,startx3,dx1,dx2,dx3;

hdf5_read_single_val(&dx1,         "/header/geom/dx1", H5T_IEEE_F64LE);
hdf5_read_single_val(&dx2,         "/header/geom/dx2", H5T_IEEE_F64LE);
hdf5_read_single_val(&dx3,         "/header/geom/dx3", H5T_IEEE_F64LE);
hdf5_read_single_val(&startx1,     "/header/geom/startx1", H5T_IEEE_F64LE);
hdf5_read_single_val(&startx2,     "/header/geom/startx2", H5T_IEEE_F64LE);
hdf5_read_single_val(&startx3,     "/header/geom/startx3", H5T_IEEE_F64LE);


printf("...dxi=%f, %f, %f\n", dx1,dx2,dx3);
printf("...startxi=%f, %f, %f\n", startx1,startx2,startx3);

if (strcmp(metric, "MMKS") == 0){
 printf("get MMKS! sorry this is not yet done....\n");
}

if (strcmp(metric, "MMKS") != 0){
//printf("MKS coordinate!! \n");
hdf5_read_single_val(&Reh,         "/header/geom/mks/r_eh", H5T_IEEE_F64LE);
hdf5_read_single_val(&Rin,         "/header/geom/mks/r_in", H5T_IEEE_F64LE);
hdf5_read_single_val(&Rout,        "/header/geom/mks/r_out", H5T_IEEE_F64LE);
hdf5_read_single_val(&a,           "/header/geom/mks/a", H5T_IEEE_F64LE);
hdf5_read_single_val(&hslope,      "/header/geom/mks/hslope", H5T_IEEE_F64LE);
}


printf("BH spin=%f!\n",a);

hsize_t fdims[] = { N1, N2, N3, N_readfrom  };
hsize_t fstart[] = { 0, 0, 0, 0 };
hsize_t fcount[] = { N1, N2, N3, N_readto};
hsize_t mdims[] = { N1, N2, N3, N_readto}; // size of data (see above)
hsize_t mstart[] = { 0, 0, 0, 0 };
hdf5_read_array(data, "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);

//hdf5_close(harmfname);
hdf5_close();
printf("===done reading...!\n");


//==assign coor related parameters

cor_set[0]=startx1;
cor_set[1]=dx1;
cor_set[2]=startx2;
cor_set[3]=dx2;
cor_set[4]=startx3;
cor_set[5]=dx3;
cor_set[6]=hslope;
cor_set[7]=a;
cor_set[8]=Reh;


num_N[0]=N1;
num_N[1]=N2;
num_N[2]=N3;
printf("startx1: %f\n",startx1);
printf("startx2: %f\n",startx2);
printf("startx3: %f\n",startx3);
printf("dx1: %f\n",dx1);
printf("dx2: %f\n",dx2);
printf("dx3: %f\n",dx3);
printf("hslope: %f\n",hslope);
int n1=0;
int n2=0;
int n3=0;
double x1 = startx1 + (n1 + cen_off) * dx1;
double x2 = startx2 + (n2 + cen_off) * dx2;
double x3 = startx3 + (n3 + cen_off) * dx3;
double r  = exp(x1); //r
double th = M_PI*x2+(1. - hslope)/2.* sin(2.* M_PI* x2); //th
double ph = x3; //ph
printf("Reh=%f Rin=%f Rout=%f\n",Reh,Rin,Rout); 
printf("start: r, th, ph = %f (Reh=%f Rin=%f Rout=%f) %f %f\n",r, Reh,Rin,Rout, th, ph);


n1=N1-1;
n2=N2-1;
n3=N3-1;
x1 = startx1 + (n1 + cen_off) * dx1;
x2 = startx2 + (n2 + cen_off) * dx2;
x3 = startx3 + (n3 + cen_off) * dx3;
r  = exp(x1); //r
th = M_PI*x2+(1. - hslope)/2.* sin(2.* M_PI* x2); //th
ph = x3; //ph
printf("end: r, th, ph = %f %f %f\n",r, th, ph);



return a;
}




