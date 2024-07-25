#include "minimizer.h"

namespace VRT2 {
double GridMinimizer::Minimize(Function& F,
			       std::vector<double>& xmin, std::vector<double>& xmax, std::vector<size_t> Nx,
			       std::vector<double>& xm)
{
  // Get total number of points for minimization
  size_t N=1;
  for (size_t i=0; i<Nx.size(); ++i)
    N *= Nx[i];

  // Do 1D->ND grid search
  std::vector<size_t> iND(Nx.size());
  std::vector<double> x(Nx.size()), im(Nx.size());
  double f, fm=0;
  for (size_t i1D=0; i1D<N; ++i1D)
  {
    iND = ND_from_1D(Nx,i1D);
    for (size_t i=0; i<Nx.size(); ++i)
      x[i] = xmin[i] + (xmax[i]-xmin[i])*double(iND[i])/double((Nx[i]>1 ? Nx[i]-1 : 1));
    f = F(x);
    if (f<fm || i1D==0)
    {
      fm = f;
      for (size_t i=0; i<Nx.size(); ++i)
      {
	im[i] = iND[i];
	xm[i] = x[i];
      }
    }

    std::cout << "TEST:"
	      << std::setw(15) << i1D;
    for (size_t i=0; i<Nx.size(); ++i)
      std::cout << std::setw(15) << iND[i];
    for (size_t i=0; i<Nx.size(); ++i)
      std::cout << std::setw(15) << x[i];
    std::cout << std::setw(15) << f
	      << std::endl;
  }

  // Now reset xmin and xmax
  for (size_t i=0; i<Nx.size(); ++i)
  {
    double dx = (xmax[i]-xmin[i])/double((Nx[i]>1 ? Nx[i]-1 : 1));
    xmin[i] = xm[i]-dx;
    xmax[i] = xm[i]+dx;
  }

  return fm;
}

std::vector<size_t> GridMinimizer::ND_from_1D(std::vector<size_t> Nx, size_t i1D) const
{
  std::vector<size_t> iND(Nx.size());

  for (size_t i=0; i<Nx.size(); ++i)
  {
    iND[i] = i1D%Nx[i];
    i1D = int(i1D/Nx[i]);
  }

  return iND;
}


double PowellMinimizer::Minimize(Function& F, std::vector<double>& xm, double tol, std::vector<double>& stps)
{
  // Define some communcation items
  _tol = tol;
  _Fptr = &F;
  _pcom = new double[_Fptr->size()];
  _xicom = new double[_Fptr->size()];
  double *p = new double[_Fptr->size()];

  // Allocate memory for direction matrix and allocate to the cardinal directions
  double **xi = new double*[_Fptr->size()];
  for (size_t i=0; i<_Fptr->size(); i++)
    xi[i] = new double[_Fptr->size()];
  for (size_t i=0; i<_Fptr->size(); i++)
    for (size_t j=0; j<_Fptr->size(); j++)
      xi[i][j] = ( i!=j ? 0.0 : stps[i] );
   
  // Fill guess
  for (size_t i=0; i<_Fptr->size(); i++)
    p[i] = xm[i];

  int iter;
  double fmin = powell(p,xi,_tol,iter);

  
  // Replace guess with minimum location
  for (size_t i=0; i<_Fptr->size(); i++)
    xm[i] = p[i];

  
  // Clean up
  for (size_t i=0; i<_Fptr->size(); i++)
    delete[] xi[i];
  delete[] xi;
  delete[] p;
  delete[] _xicom;
  delete[] _pcom;

  // Return minimum value
  return fmin;
}

double PowellMinimizer::f1dim(double x)
{
  std::vector<double> xt(_Fptr->size());
  for (size_t j=0; j<_Fptr->size(); j++)
    xt[j]=_pcom[j]+x*_xicom[j];

  return (*_Fptr)(xt);
}

double PowellMinimizer::linmin(double p[], double xi[])
{
  
  for (size_t j=0;j<_Fptr->size();j++) {
    _pcom[j]=p[j];
    _xicom[j]=xi[j];
  }
  double ax=0.0;
  double xx=1.0;
  double xmin,fx,fb,fa,bx;
  mnbrak(ax,xx,bx,fa,fx,fb);
  double fret=brent(ax,xx,bx,_tol,xmin);
  for (size_t j=0;j<_Fptr->size();j++) {
    xi[j] *= xmin;
    p[j] += xi[j];
  }
  return fret;
}

#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
#define SIGN(a,b) ((b) >= 0.0 ? std::fabs(a) : -std::fabs(a))
void PowellMinimizer::mnbrak(double& ax, double& bx, double& cx, double& fa, double& fb, double& fc)
{
  double ulim,u,r,q,fu,dum;

  fa=f1dim(ax);
  fb=f1dim(bx);
  if (fb > fa) {
    SHFT(dum,ax,bx,dum);
    SHFT(dum,fb,fa,dum);
  }
  cx=bx+GOLD*(bx-ax);
  fc=f1dim(cx);

  while (fb > fc) {

    r=(bx-ax)*(fb-fc);
    q=(bx-cx)*(fb-fa);
    u=bx-((bx-cx)*q-(bx-ax)*r)/(2.0*SIGN(std::max(std::fabs(q-r),TINY),q-r));
    ulim=bx+GLIMIT*(cx-bx);

    if ((bx-u)*(u-cx) > 0.0) {
      fu=f1dim(u);
      if (fu < fc) {
	ax=bx;
	bx=u;
	fa=fb;
	fb=fu;
	return;
      } else if (fu > fb) {
	cx=u;
	fc=fu;
	return;
      }
      u=cx+GOLD*(cx-bx);
      fu=f1dim(u);
    } else if ((cx-u)*(u-ulim) > 0.0) {
      fu=f1dim(u);
      if (fu < fc) {
	SHFT(bx,cx,u,cx+GOLD*(cx-bx));
	SHFT(fb,fc,fu,f1dim(u));
      }
    } else if ((u-ulim)*(ulim-cx) >= 0.0) {
      u=ulim;
      fu=f1dim(u);
    } else {
      u=(cx)+GOLD*(cx-bx);
      fu=f1dim(u);
    }
    SHFT(ax,bx,cx,u);
    SHFT(fa,fb,fc,fu);
  }
}
#undef GOLD
#undef GLIMIT
#undef TINY
#undef SHFT
#undef SIGN

#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
#define SIGN(a,b) ((b) >= 0.0 ? std::fabs(a) : -std::fabs(a))
double PowellMinimizer::brent(double ax, double bx, double cx, double tol, double& xmin)
{
  int iter;
  double a,b,d=0,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
  double e=0.0;
  
  a=(ax < cx ? ax : cx);
  b=(ax > cx ? ax : cx);
  x=w=v=bx;
  fw=fv=fx=f1dim(x);
  for (iter=1;iter<=ITMAX;iter++) {
    xm=0.5*(a+b);
    tol2=2.0*(tol1=tol*std::fabs(x)+ZEPS);
    if (std::fabs(x-xm) <= (tol2-0.5*(b-a))) {
      xmin=x;
      return fx;
    }
    if (std::fabs(e) > tol1) {
      r=(x-w)*(fx-fv);
      q=(x-v)*(fx-fw);
      p=(x-v)*q-(x-w)*r;
      q=2.0*(q-r);
      if (q > 0.0) p = -p;
      q=std::fabs(q);
      etemp=e;
      e=d;
      if (std::fabs(p) >= std::fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
	d=CGOLD*(e=(x >= xm ? a-x : b-x));
      else {
	d=p/q;
	u=x+d;
	if (u-a < tol2 || b-u < tol2)
	  d=SIGN(tol1,xm-x);
      }
    } else {
      d=CGOLD*(e=(x >= xm ? a-x : b-x));
    }
    u=(std::fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
    fu=f1dim(u);
    if (fu <= fx) {
      if (u >= x) a=x; else b=x;
      SHFT(v,w,x,u);
      SHFT(fv,fw,fx,fu);
    } else {
      if (u < x) a=u; else b=u;
      if (fu <= fw || w == x) {
	v=w;
	w=u;
	fv=fw;
	fw=fu;
      } else if (fu <= fv || v == x || v == w) {
	v=u;
	fv=fu;
      }
    }
  }
  nrerror("Too many iterations in brent");
  xmin=x;
  return fx;
}
#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SHFT
#undef SIGN

#define ITMAX 200
double PowellMinimizer::powell(double p[], double **xi, double ftol, int& iter)
{
  size_t ibig;
  double del,fp,fptt,t,*pt,*ptt,*xit;
  double fret;

  pt = new double[_Fptr->size()];
  ptt = new double[_Fptr->size()];
  xit = new double[_Fptr->size()];

  std::vector<double> pv(_Fptr->size());
  for (size_t i=0; i<_Fptr->size(); i++)
    pv[i] = p[i];
  fret=(*_Fptr)(pv);

  for (size_t j=0;j<_Fptr->size();j++)
    pt[j]=p[j];

  for (iter=1;;++iter) {

    fp=fret;
    ibig=0;
    del=0.0;

    for (size_t i=0;i<_Fptr->size();i++) {
      for (size_t j=0;j<_Fptr->size();j++)
	xit[j]=xi[j][i];
      fptt=fret;

      fret = linmin(p,xit);
      if (std::fabs(fptt-fret) > del) {
	del=std::fabs(fptt-fret);
	ibig=i;
      }
    }
    if (2.0*std::fabs(fp-fret) <= ftol*(std::fabs(fp)+std::fabs(fret))) {
      delete[] pt;
      delete[] ptt;
      delete[] xit;
      return fret;
    }
    if (iter == ITMAX) {
      delete[] pt;
      delete[] ptt;
      delete[] xit;
      nrerror("powell exceeding maximum iterations.");
    }
    for (size_t j=0;j<_Fptr->size();j++) {
      ptt[j]=2.0*p[j]-pt[j];
      xit[j]=p[j]-pt[j];
      pt[j]=p[j];
    }
    for (size_t i=0; i<_Fptr->size(); i++)
      pv[i] = ptt[i];
    fptt=(*_Fptr)(pv);
    if (fptt < fp) {
      t=2.0*(fp-2.0*fret+fptt)*(fp-fret-del)*(fp-fret-del)-del*(fp-fptt)*(fp-fptt);
      if (t < 0.0) {
	fret = linmin(p,xit);
	for (size_t j=0;j<_Fptr->size();j++) {
	  xi[j][ibig]=xi[j][_Fptr->size()-1];
	  xi[j][_Fptr->size()-1]=xit[j];
	}
      }
    }
  }
}
#undef ITMAX

};

