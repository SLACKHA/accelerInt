////////////////////////////////////////////////////////////////////////////////
// File: embedded_fehlberg_7_8.c                                              //
// Routines:                                                                  //
//    Embedded_Fehlberg_7_8                                                   //
// Adapted from: http://www.mymathlib.com/c_source/                           //
// diffeq/embedded_runge_kutta/embedded_fehlberg_7_8.c                        //
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  Description:                                                              //
//     The Runge-Kutta-Fehlberg method is an adaptive procedure for approxi-  //
//     mating the solution of the differential equation y'(x) = f(x,y) with   //
//     initial condition y(x0) = c.  This implementation evaluates f(x,y)     //
//     thirteen times per step using embedded seventh order and eight order   //
//     Runge-Kutta estimates to estimate the not only the solution but also   //
//     the error.                                                             //
//     The next step size is then calculated using the preassigned tolerance  //
//     and error estimate.                                                    //
//     For step i+1,                                                          //
//        y[i+1] = y[i] +  h * (41/840 * k1 + 34/105 * k6 + 9/35 * k7         //
//                        + 9/35 * k8 + 9/280 * k9 + 9/280 k10 + 41/840 k11 ) //
//     where                                                                  //
//     k1 = f( x[i],y[i] ),                                                   //
//     k2 = f( x[i]+2h/27, y[i] + 2h*k1/27),                                  //
//     k3 = f( x[i]+h/9, y[i]+h/36*( k1 + 3 k2) ),                            //
//     k4 = f( x[i]+h/6, y[i]+h/24*( k1 + 3 k3) ),                            //
//     k5 = f( x[i]+5h/12, y[i]+h/48*(20 k1 - 75 k3 + 75 k4)),                //
//     k6 = f( x[i]+h/2, y[i]+h/20*( k1 + 5 k4 + 4 k5 ) ),                    //
//     k7 = f( x[i]+5h/6, y[i]+h/108*( -25 k1 + 125 k4 - 260 k5 + 250 k6 ) ), //
//     k8 = f( x[i]+h/6, y[i]+h*( 31/300 k1 + 61/225 k5 - 2/9 k6              //
//                                                            + 13/900 K7) )  //
//     k9 = f( x[i]+2h/3, y[i]+h*( 2 k1 - 53/6 k4 + 704/45 k5 - 107/9 k6      //
//                                                      + 67/90 k7 + 3 k8) ), //
//     k10 = f( x[i]+h/3, y[i]+h*( -91/108 k1 + 23/108 k4 - 976/135 k5        //
//                             + 311/54 k6 - 19/60 k7 + 17/6 K8 - 1/12 k9) ), //
//     k11 = f( x[i]+h, y[i]+h*( 2383/4100 k1 - 341/164 k4 + 4496/1025 k5     //
//          - 301/82 k6 + 2133/4100 k7 + 45/82 K8 + 45/164 k9 + 18/41 k10) )  //
//     k12 = f( x[i], y[i]+h*( 3/205 k1 - 6/41 k6 - 3/205 k7 - 3/41 K8        //
//                                                   + 3/41 k9 + 6/41 k10) )  //
//     k13 = f( x[i]+h, y[i]+h*( -1777/4100 k1 - 341/164 k4 + 4496/1025 k5    //
//                      - 289/82 k6 + 2193/4100 k7 + 51/82 K8 + 33/164 k9 +   //
//                                                        12/41 k10 + k12) )  //
//     x[i+1] = x[i] + h.                                                     //
//                                                                            //
//     The error is estimated to be                                           //
//        err = -41/840 * h * ( k1 + k11 - k12 - k13)                         //
//     The step size h is then scaled by the scale factor                     //
//         scale = 0.8 * | epsilon * y[i] / [err * (xmax - x[0])] | ^ 1/7     //
//     The scale factor is further constrained 0.125 < scale < 4.0.           //
//     The new step size is h := scale * h.                                   //
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <string.h>

#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )

#include "header.h"
#include "dydt.h"

static void Runge_Kutta(const double t0, const double* y0, const double pr, const double h,
                          double* y1, double* err);

////////////////////////////////////////////////////////////////////////////////
// int Embedded_Fehlberg_7_8( double (*f)(double, double), double y[],        //
//       double x, double h, double xmax, double *h_next, double tolerance )  //
//                                                                            //
//  Description:                                                              //
//     This function solves the differential equation y'=f(x,y) with the      //
//     initial condition y(x) = y[0].  The value at xmax is returned in y[1]. //
//     The function returns 0 if successful or -1 if it fails.                //
//                                                                            //
//  Arguments:                                                                //
//     double *f  Pointer to the function which returns the slope at (x,y) of //
//                integral curve of the differential equation y' = f(x,y)     //
//                which passes through the point (x0,y0) corresponding to the //
//                initial condition y(x0) = y0.                               //
//     double y[] On input y[0] is the initial value of y at x, on output     //
//                y[1] is the solution at xmax.                               //
//     double x   The initial value of x.                                     //
//     double h   Initial step size.                                          //
//     double xmax The endpoint of x.                                         //
//     double *h_next   A pointer to the estimated step size for successive   //
//                      calls to Embedded_Fehlberg_7_8.                       //
//     double tolerance The tolerance of y(xmax), i.e. a solution is sought   //
//                so that the relative error < tolerance.                     //
//                                                                            //
//  Return Values:                                                            //
//     0   The solution of y' = f(x,y) from x to xmax is stored y[1] and      //
//         h_next has the value to the next size to try.                      //
//    -1   The solution of y' = f(x,y) from x to xmax failed.                 //
//    -2   Failed because either xmax < x or the step size h <= 0.            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int integrate(const double t_start, const double t_end, const double pr, double* y) {

   static const double err_exponent = 1.0 / 7.0;

   double t = t_end;
   double y0[NSP];
   memcpy(y0, y, NSP * sizeof(double));
   double err[NSP];
   double h = t_end - t_start;

   Runge_Kutta(t_start, y0, pr, h, y, err);
   //scale(err);

   return 0;
}


////////////////////////////////////////////////////////////////////////////////
//  static double Runge_Kutta(double (*f)(double,double), double *y,          //
//                                                       double x0, double h) //
//                                                                            //
//  Description:                                                              //
//     This routine uses Fehlberg's embedded 7th and 8th order methods to     //
//     approximate the solution of the differential equation y'=f(x,y) with   //
//     the initial condition y = y[0] at x = x0.  The value at x + h is       //
//     returned in y[1].  The function returns err / h ( the absolute error   //
//     per step size ).                                                       //
//                                                                            //
//  Arguments:                                                                //
//     double *f  Pointer to the function which returns the slope at (x,y) of //
//                integral curve of the differential equation y' = f(x,y)     //
//                which passes through the point (x0,y[0]).                   //
//     double y[] On input y[0] is the initial value of y at x, on output     //
//                y[1] is the solution at x + h.                              //
//     double x   Initial value of x.                                         //
//     double h   Step size                                                   //
//                                                                            //
//  Return Values:                                                            //
//     This routine returns the err / h.  The solution of y(x) at x + h is    //
//     returned in y[1].                                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

static void Runge_Kutta(const double t0, const double* y0, const double pr, const double h,
                          double* y1, double* err) {
   
   static const double c_1_11 = 41.0 / 840.0;
   static const double c_6 = 34.0 / 105.0;
   static const double c_7_8= 9.0 / 35.0;
   static const double c_9_10 = 9.0 / 280.0;

   static const double a2 = 2.0 / 27.0;
   static const double a3 = 1.0 / 9.0;
   static const double a4 = 1.0 / 6.0;
   static const double a5 = 5.0 / 12.0;
   static const double a6 = 1.0 / 2.0;
   static const double a7 = 5.0 / 6.0;
   static const double a8 = 1.0 / 6.0;
   static const double a9 = 2.0 / 3.0;
   static const double a10 = 1.0 / 3.0;

   static const double b31 = 1.0 / 36.0;
   static const double b32 = 3.0 / 36.0;
   static const double b41 = 1.0 / 24.0;
   static const double b43 = 3.0 / 24.0;
   static const double b51 = 20.0 / 48.0;
   static const double b53 = -75.0 / 48.0;
   static const double b54 = 75.0 / 48.0;
   static const double b61 = 1.0 / 20.0;
   static const double b64 = 5.0 / 20.0;
   static const double b65 = 4.0 / 20.0;
   static const double b71 = -25.0 / 108.0;
   static const double b74 =  125.0 / 108.0;
   static const double b75 = -260.0 / 108.0;
   static const double b76 =  250.0 / 108.0;
   static const double b81 = 31.0/300.0;
   static const double b85 = 61.0/225.0;
   static const double b86 = -2.0/9.0;
   static const double b87 = 13.0/900.0;
   static const double b91 = 2.0;
   static const double b94 = -53.0/6.0;
   static const double b95 = 704.0 / 45.0;
   static const double b96 = -107.0 / 9.0;
   static const double b97 = 67.0 / 90.0;
   static const double b98 = 3.0;
   static const double b10_1 = -91.0 / 108.0;
   static const double b10_4 = 23.0 / 108.0;
   static const double b10_5 = -976.0 / 135.0;
   static const double b10_6 = 311.0 / 54.0;
   static const double b10_7 = -19.0 / 60.0;
   static const double b10_8 = 17.0 / 6.0;
   static const double b10_9 = -1.0 / 12.0;
   static const double b11_1 = 2383.0 / 4100.0;
   static const double b11_4 = -341.0 / 164.0;
   static const double b11_5 = 4496.0 / 1025.0;
   static const double b11_6 = -301.0 / 82.0;
   static const double b11_7 = 2133.0 / 4100.0;
   static const double b11_8 = 45.0 / 82.0;
   static const double b11_9 = 45.0 / 164.0;
   static const double b11_10 = 18.0 / 41.0;
   static const double b12_1 = 3.0 / 205.0;
   static const double b12_6 = - 6.0 / 41.0;
   static const double b12_7 = - 3.0 / 205.0;
   static const double b12_8 = - 3.0 / 41.0;
   static const double b12_9 = 3.0 / 41.0;
   static const double b12_10 = 6.0 / 41.0;
   static const double b13_1 = -1777.0 / 4100.0;
   static const double b13_4 = -341.0 / 164.0;
   static const double b13_5 = 4496.0 / 1025.0;
   static const double b13_6 = -289.0 / 82.0;
   static const double b13_7 = 2193.0 / 4100.0;
   static const double b13_8 = 51.0 / 82.0;
   static const double b13_9 = 33.0 / 164.0;
   static const double b13_10 = 12.0 / 41.0;
   
   static const double err_factor  = -41.0 / 840.0;

   double k1[NSP] = {0};
   double k2[NSP] = {0};
   double k3[NSP] = {0};
   double k4[NSP] = {0};
   double k5[NSP] = {0};
   double k6[NSP] = {0};
   double k7[NSP] = {0};
   double k8[NSP] = {0};
   double k9[NSP] = {0};
   double k10[NSP] = {0};
   double k11[NSP] = {0};
   double k12[NSP] = {0};
   double k13[NSP] = {0};
   double h2_7 = a2 * h;

   double y[NSP] = {0};
   memcpy(y, y0, NSP * sizeof(double));

   dydt(t0, pr, y, k1);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h2_7 * k1[i];
  }
   dydt(t0+h2_7, pr, y, k2);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b31 * k1[i] + b32 * k2[i]);
  }
  dydt(t0+a3*h, pr, y, k3);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b41 * k1[i] + b43 * k3[i]);
  }
  dydt(t0+a4*h, pr, y, k4);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b51 * k1[i] + b53 * k3[i] + b54 * k4[i]);
  }
  dydt(t0+a5*h, pr, y, k5);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b61 * k1[i] + b64 * k4[i] + b65 * k5[i]);
  }
  dydt(t0+a6*h, pr, y, k6);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b71 * k1[i] + b74 * k4[i] + b75 * k5[i] + b76 * k6[i]);
  }
  dydt(t0+a7*h, pr, y, k7);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b81 * k1[i] + b85 * k5[i] + b86 * k6[i] + b87 * k7[i]);
  }
  dydt(t0+a8*h, pr, y, k8);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b91 * k1[i] + b94 * k4[i] + b95 * k5[i] + b96 * k6[i] + 
                            b97 * k7[i] + b98 * k8[i]);
  }
  dydt(t0+a9*h, pr, y, k9);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b10_1 * k1[i] + b10_4 * k4[i] + b10_5 * k5[i] + b10_6 * k6[i] + 
                            b10_7 * k7[i] + b10_8 * k8[i] + b10_9 * k9[i]);
  }
  dydt(t0+a10*h, pr, y, k10);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b11_1 * k1[i] + b11_4 * k4[i] + b11_5 * k5[i] + b11_6 * k6[i] + 
                            b11_7 * k7[i] + b11_8 * k8[i] + b11_9 * k9[i] + b11_10 * k10[i]);
  }
  dydt(t0+h, pr, y, k11);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b12_1 * k1[i] + b12_6 * k6[i] + b12_7 * k7[i] + b12_8 * k8[i] + 
        b12_9 * k9[i] + b12_10 * k10[i]);
  }
  dydt(t0, pr, y, k12);
  for(int i = 0; i < NSP; ++i)
  {
     y[i] = y0[i] + h * (b13_1 * k1[i] + b13_4 * k4[i] + b13_5 * k5[i] + b13_6 * k6[i] + 
                            b13_7 * k7[i] + b13_8 * k8[i] + b13_9 * k9[i] + b13_10 * k10[i] + k12[i]);
  }

  //compute y1 and error
  for (int i = 0; i < NSP; ++i)
  {
      y1[i] = y0[i] + h * (c_1_11 * (k1[i] + k11[i]) + c_6 * k6[i] + 
                            c_7_8 * (k7[i] + k8[i]) + c_9_10 * (k9[i] + k10[i]));
      err[i] = err_factor * (k1[i] + k11[i] - k12[i] - k13[i]);
  }
}

 void cleanup_solver(int num_threads) {
 }

 const char* solver_name() {
  const char* name = "rk78-int";
  return name;
 }

 void initialize_solver(int num_threads) {

 }

 void init_solver_log() {
  
 }
 void solver_log() {
  
 }