#include "head.h"

__device__ void eval_h (const Real T, Real * h) {

  if (T <= 1000.0) {
    h[0] = 8.24876732e+07 * (2.54716300e+04 + T * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  } else {
    h[0] = 8.24876732e+07 * (2.54716300e+04 + T * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  }

  if (T <= 1000.0) {
    h[1] = 4.12438366e+07 * (-1.01252100e+03 + T * (3.29812400e+00 + T * (4.12472100e-04 + T * (-2.71433833e-07 + T * (-2.36885850e-11 + 8.26974400e-14 * T)))));
  } else {
    h[1] = 4.12438366e+07 * (-8.35034000e+02 + T * (2.99142300e+00 + T * (3.50032200e-04 + T * (-1.87794300e-08 + T * (-2.30789450e-12 + 3.16550400e-16 * T)))));
  }

  if (T <= 1000.0) {
    h[2] = 5.19676363e+06 * (2.91476400e+04 + T * (2.94642900e+00 + T * (-8.19083000e-04 + T * (8.07010667e-07 + T * (-4.00710750e-10 + 7.78139200e-14 * T)))));
  } else {
    h[2] = 5.19676363e+06 * (2.92308000e+04 + T * (2.54206000e+00 + T * (-1.37753100e-05 + T * (-1.03426767e-09 + T * (1.13776675e-12 + -8.73610400e-17 * T)))));
  }

  if (T <= 1000.0) {
    h[3] = 4.88876881e+06 * (3.34630913e+03 + T * (4.12530561e+00 + T * (-1.61272470e-03 + T * (2.17588230e-06 + T * (-1.44963411e-09 + 4.12474758e-13 * T)))));
  } else {
    h[3] = 4.88876881e+06 * (3.68362875e+03 + T * (2.86472886e+00 + T * (5.28252240e-04 + T * (-8.63609193e-08 + T * (7.63046685e-12 + -2.66391752e-16 * T)))));
  }

  if (T <= 1000.0) {
    h[4] = 4.61523901e+06 * (-3.02081100e+04 + T * (3.38684200e+00 + T * (1.73749100e-03 + T * (-2.11823200e-06 + T * (1.74214525e-09 + -5.01317600e-13 * T)))));
  } else {
    h[4] = 4.61523901e+06 * (-2.98992100e+04 + T * (2.67214600e+00 + T * (1.52814650e-03 + T * (-2.91008667e-07 + T * (3.00249000e-11 + -1.27832360e-15 * T)))));
  }

  if (T <= 1000.0) {
    h[5] = 2.59838181e+06 * (-1.00524900e+03 + T * (3.21293600e+00 + T * (5.63743000e-04 + T * (-1.91871667e-07 + T * (3.28469250e-10 + -1.75371080e-13 * T)))));
  } else {
    h[5] = 2.59838181e+06 * (-1.23393000e+03 + T * (3.69757800e+00 + T * (3.06759850e-04 + T * (-4.19614000e-08 + T * (4.43820250e-12 + -2.27287000e-16 * T)))));
  }

  if (T <= 1000.0) {
    h[6] = 2.51903170e+06 * (2.94808040e+02 + T * (4.30179801e+00 + T * (-2.37456025e-03 + T * (7.05276303e-06 + T * (-6.06909735e-09 + 1.85845025e-12 * T)))));
  } else {
    h[6] = 2.51903170e+06 * (1.11856713e+02 + T * (4.01721090e+00 + T * (1.11991006e-03 + T * (-2.11219383e-07 + T * (2.85615925e-11 + -2.15817070e-15 * T)))));
  }

  if (T <= 1000.0) {
    h[7] = 2.44438441e+06 * (-1.76631500e+04 + T * (3.38875400e+00 + T * (3.28461300e-03 + T * (-4.95004333e-08 + T * (-1.15645150e-09 + 4.94303000e-13 * T)))));
  } else {
    h[7] = 2.44438441e+06 * (-1.80069600e+04 + T * (4.57316700e+00 + T * (2.16806800e-03 + T * (-4.91563000e-07 + T * (5.87226000e-11 + -2.86330800e-15 * T)))));
  }

  if (T <= 1000.0) {
    h[8] = 2.96804743e+06 * (-1.02090000e+03 + T * (3.29867700e+00 + T * (7.04120000e-04 + T * (-1.32107400e-06 + T * (1.41037875e-09 + -4.88971000e-13 * T)))));
  } else {
    h[8] = 2.96804743e+06 * (-9.22797700e+02 + T * (2.92664000e+00 + T * (7.43988500e-04 + T * (-1.89492033e-07 + T * (2.52426000e-11 + -1.35067020e-15 * T)))));
  }

  if (T <= 1000.0) {
    h[9] = 2.08133323e+06 * (-7.45375000e+02 + T * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  } else {
    h[9] = 2.08133323e+06 * (-7.45375000e+02 + T * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  }

  if (T <= 1000.0) {
    h[10] = 2.07727727e+07 * (-7.45375000e+02 + T * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  } else {
    h[10] = 2.07727727e+07 * (-7.45375000e+02 + T * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  }

  if (T <= 1000.0) {
    h[11] = 2.96834943e+06 * (-1.43105400e+04 + T * (3.26245200e+00 + T * (7.55970500e-04 + T * (-1.29391833e-06 + T * (1.39548600e-09 + -4.94990200e-13 * T)))));
  } else {
    h[11] = 2.96834943e+06 * (-1.42683500e+04 + T * (3.02507800e+00 + T * (7.21344500e-04 + T * (-1.87694267e-07 + T * (2.54645250e-11 + -1.38219040e-15 * T)))));
  }

  if (T <= 1000.0) {
    h[12] = 1.88923414e+06 * (-4.83731400e+04 + T * (2.27572500e+00 + T * (4.96103600e-03 + T * (-3.46970333e-06 + T * (1.71667175e-09 + -4.23456000e-13 * T)))));
  } else {
    h[12] = 1.88923414e+06 * (-4.89669600e+04 + T * (4.45362300e+00 + T * (1.57008450e-03 + T * (-4.26137000e-07 + T * (5.98499250e-11 + -3.33806600e-15 * T)))));
  }

} // end eval_h

__device__ void eval_u (const Real T, Real * u) {

  if (T <= 1000.0) {
    u[0] = 8.24876732e+07 * (2.54716300e+04 + T * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  } else {
    u[0] = 8.24876732e+07 * (2.54716300e+04 + T * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  }

  if (T <= 1000.0) {
    u[1] = 4.12438366e+07 * (-1.01252100e+03 + T * (3.29812400e+00 - 1.0 + T * (4.12472100e-04 + T * (-2.71433833e-07 + T * (-2.36885850e-11 + 8.26974400e-14 * T)))));
  } else {
    u[1] = 4.12438366e+07 * (-8.35034000e+02 + T * (2.99142300e+00 - 1.0 + T * (3.50032200e-04 + T * (-1.87794300e-08 + T * (-2.30789450e-12 + 3.16550400e-16 * T)))));
  }

  if (T <= 1000.0) {
    u[2] = 5.19676363e+06 * (2.91476400e+04 + T * (2.94642900e+00 - 1.0 + T * (-8.19083000e-04 + T * (8.07010667e-07 + T * (-4.00710750e-10 + 7.78139200e-14 * T)))));
  } else {
    u[2] = 5.19676363e+06 * (2.92308000e+04 + T * (2.54206000e+00 - 1.0 + T * (-1.37753100e-05 + T * (-1.03426767e-09 + T * (1.13776675e-12 + -8.73610400e-17 * T)))));
  }

  if (T <= 1000.0) {
    u[3] = 4.88876881e+06 * (3.34630913e+03 + T * (4.12530561e+00 - 1.0 + T * (-1.61272470e-03 + T * (2.17588230e-06 + T * (-1.44963411e-09 + 4.12474758e-13 * T)))));
  } else {
    u[3] = 4.88876881e+06 * (3.68362875e+03 + T * (2.86472886e+00 - 1.0 + T * (5.28252240e-04 + T * (-8.63609193e-08 + T * (7.63046685e-12 + -2.66391752e-16 * T)))));
  }

  if (T <= 1000.0) {
    u[4] = 4.61523901e+06 * (-3.02081100e+04 + T * (3.38684200e+00 - 1.0 + T * (1.73749100e-03 + T * (-2.11823200e-06 + T * (1.74214525e-09 + -5.01317600e-13 * T)))));
  } else {
    u[4] = 4.61523901e+06 * (-2.98992100e+04 + T * (2.67214600e+00 - 1.0 + T * (1.52814650e-03 + T * (-2.91008667e-07 + T * (3.00249000e-11 + -1.27832360e-15 * T)))));
  }

  if (T <= 1000.0) {
    u[5] = 2.59838181e+06 * (-1.00524900e+03 + T * (3.21293600e+00 - 1.0 + T * (5.63743000e-04 + T * (-1.91871667e-07 + T * (3.28469250e-10 + -1.75371080e-13 * T)))));
  } else {
    u[5] = 2.59838181e+06 * (-1.23393000e+03 + T * (3.69757800e+00 - 1.0 + T * (3.06759850e-04 + T * (-4.19614000e-08 + T * (4.43820250e-12 + -2.27287000e-16 * T)))));
  }

  if (T <= 1000.0) {
    u[6] = 2.51903170e+06 * (2.94808040e+02 + T * (4.30179801e+00 - 1.0 + T * (-2.37456025e-03 + T * (7.05276303e-06 + T * (-6.06909735e-09 + 1.85845025e-12 * T)))));
  } else {
    u[6] = 2.51903170e+06 * (1.11856713e+02 + T * (4.01721090e+00 - 1.0 + T * (1.11991006e-03 + T * (-2.11219383e-07 + T * (2.85615925e-11 + -2.15817070e-15 * T)))));
  }

  if (T <= 1000.0) {
    u[7] = 2.44438441e+06 * (-1.76631500e+04 + T * (3.38875400e+00 - 1.0 + T * (3.28461300e-03 + T * (-4.95004333e-08 + T * (-1.15645150e-09 + 4.94303000e-13 * T)))));
  } else {
    u[7] = 2.44438441e+06 * (-1.80069600e+04 + T * (4.57316700e+00 - 1.0 + T * (2.16806800e-03 + T * (-4.91563000e-07 + T * (5.87226000e-11 + -2.86330800e-15 * T)))));
  }

  if (T <= 1000.0) {
    u[8] = 2.96804743e+06 * (-1.02090000e+03 + T * (3.29867700e+00 - 1.0 + T * (7.04120000e-04 + T * (-1.32107400e-06 + T * (1.41037875e-09 + -4.88971000e-13 * T)))));
  } else {
    u[8] = 2.96804743e+06 * (-9.22797700e+02 + T * (2.92664000e+00 - 1.0 + T * (7.43988500e-04 + T * (-1.89492033e-07 + T * (2.52426000e-11 + -1.35067020e-15 * T)))));
  }

  if (T <= 1000.0) {
    u[9] = 2.08133323e+06 * (-7.45375000e+02 + T * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  } else {
    u[9] = 2.08133323e+06 * (-7.45375000e+02 + T * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  }

  if (T <= 1000.0) {
    u[10] = 2.07727727e+07 * (-7.45375000e+02 + T * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  } else {
    u[10] = 2.07727727e+07 * (-7.45375000e+02 + T * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T)))));
  }

  if (T <= 1000.0) {
    u[11] = 2.96834943e+06 * (-1.43105400e+04 + T * (3.26245200e+00 - 1.0 + T * (7.55970500e-04 + T * (-1.29391833e-06 + T * (1.39548600e-09 + -4.94990200e-13 * T)))));
  } else {
    u[11] = 2.96834943e+06 * (-1.42683500e+04 + T * (3.02507800e+00 - 1.0 + T * (7.21344500e-04 + T * (-1.87694267e-07 + T * (2.54645250e-11 + -1.38219040e-15 * T)))));
  }

  if (T <= 1000.0) {
    u[12] = 1.88923414e+06 * (-4.83731400e+04 + T * (2.27572500e+00 - 1.0 + T * (4.96103600e-03 + T * (-3.46970333e-06 + T * (1.71667175e-09 + -4.23456000e-13 * T)))));
  } else {
    u[12] = 1.88923414e+06 * (-4.89669600e+04 + T * (4.45362300e+00 - 1.0 + T * (1.57008450e-03 + T * (-4.26137000e-07 + T * (5.98499250e-11 + -3.33806600e-15 * T)))));
  }

} // end eval_u

__device__ void eval_cv (const Real T, Real * cv) {

  if (T <= 1000.0) {
    cv[0] = 8.24876732e+07 * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  } else {
    cv[0] = 8.24876732e+07 * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  }

  if (T <= 1000.0) {
    cv[1] = 4.12438366e+07 * (3.29812400e+00 - 1.0 + T * (8.24944200e-04 + T * (-8.14301500e-07 + T * (-9.47543400e-11 + 4.13487200e-13 * T))));
  } else {
    cv[1] = 4.12438366e+07 * (2.99142300e+00 - 1.0 + T * (7.00064400e-04 + T * (-5.63382900e-08 + T * (-9.23157800e-12 + 1.58275200e-15 * T))));
  }

  if (T <= 1000.0) {
    cv[2] = 5.19676363e+06 * (2.94642900e+00 - 1.0 + T * (-1.63816600e-03 + T * (2.42103200e-06 + T * (-1.60284300e-09 + 3.89069600e-13 * T))));
  } else {
    cv[2] = 5.19676363e+06 * (2.54206000e+00 - 1.0 + T * (-2.75506200e-05 + T * (-3.10280300e-09 + T * (4.55106700e-12 + -4.36805200e-16 * T))));
  }

  if (T <= 1000.0) {
    cv[3] = 4.88876881e+06 * (4.12530561e+00 - 1.0 + T * (-3.22544939e-03 + T * (6.52764691e-06 + T * (-5.79853643e-09 + 2.06237379e-12 * T))));
  } else {
    cv[3] = 4.88876881e+06 * (2.86472886e+00 - 1.0 + T * (1.05650448e-03 + T * (-2.59082758e-07 + T * (3.05218674e-11 + -1.33195876e-15 * T))));
  }

  if (T <= 1000.0) {
    cv[4] = 4.61523901e+06 * (3.38684200e+00 - 1.0 + T * (3.47498200e-03 + T * (-6.35469600e-06 + T * (6.96858100e-09 + -2.50658800e-12 * T))));
  } else {
    cv[4] = 4.61523901e+06 * (2.67214600e+00 - 1.0 + T * (3.05629300e-03 + T * (-8.73026000e-07 + T * (1.20099600e-10 + -6.39161800e-15 * T))));
  }

  if (T <= 1000.0) {
    cv[5] = 2.59838181e+06 * (3.21293600e+00 - 1.0 + T * (1.12748600e-03 + T * (-5.75615000e-07 + T * (1.31387700e-09 + -8.76855400e-13 * T))));
  } else {
    cv[5] = 2.59838181e+06 * (3.69757800e+00 - 1.0 + T * (6.13519700e-04 + T * (-1.25884200e-07 + T * (1.77528100e-11 + -1.13643500e-15 * T))));
  }

  if (T <= 1000.0) {
    cv[6] = 2.51903170e+06 * (4.30179801e+00 - 1.0 + T * (-4.74912051e-03 + T * (2.11582891e-05 + T * (-2.42763894e-08 + 9.29225124e-12 * T))));
  } else {
    cv[6] = 2.51903170e+06 * (4.01721090e+00 - 1.0 + T * (2.23982013e-03 + T * (-6.33658150e-07 + T * (1.14246370e-10 + -1.07908535e-14 * T))));
  }

  if (T <= 1000.0) {
    cv[7] = 2.44438441e+06 * (3.38875400e+00 - 1.0 + T * (6.56922600e-03 + T * (-1.48501300e-07 + T * (-4.62580600e-09 + 2.47151500e-12 * T))));
  } else {
    cv[7] = 2.44438441e+06 * (4.57316700e+00 - 1.0 + T * (4.33613600e-03 + T * (-1.47468900e-06 + T * (2.34890400e-10 + -1.43165400e-14 * T))));
  }

  if (T <= 1000.0) {
    cv[8] = 2.96804743e+06 * (3.29867700e+00 - 1.0 + T * (1.40824000e-03 + T * (-3.96322200e-06 + T * (5.64151500e-09 + -2.44485500e-12 * T))));
  } else {
    cv[8] = 2.96804743e+06 * (2.92664000e+00 - 1.0 + T * (1.48797700e-03 + T * (-5.68476100e-07 + T * (1.00970400e-10 + -6.75335100e-15 * T))));
  }

  if (T <= 1000.0) {
    cv[9] = 2.08133323e+06 * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  } else {
    cv[9] = 2.08133323e+06 * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  }

  if (T <= 1000.0) {
    cv[10] = 2.07727727e+07 * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  } else {
    cv[10] = 2.07727727e+07 * (2.50000000e+00 - 1.0 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  }

  if (T <= 1000.0) {
    cv[11] = 2.96834943e+06 * (3.26245200e+00 - 1.0 + T * (1.51194100e-03 + T * (-3.88175500e-06 + T * (5.58194400e-09 + -2.47495100e-12 * T))));
  } else {
    cv[11] = 2.96834943e+06 * (3.02507800e+00 - 1.0 + T * (1.44268900e-03 + T * (-5.63082800e-07 + T * (1.01858100e-10 + -6.91095200e-15 * T))));
  }

  if (T <= 1000.0) {
    cv[12] = 1.88923414e+06 * (2.27572500e+00 - 1.0 + T * (9.92207200e-03 + T * (-1.04091100e-05 + T * (6.86668700e-09 + -2.11728000e-12 * T))));
  } else {
    cv[12] = 1.88923414e+06 * (4.45362300e+00 - 1.0 + T * (3.14016900e-03 + T * (-1.27841100e-06 + T * (2.39399700e-10 + -1.66903300e-14 * T))));
  }

} // end eval_cv

__device__ void eval_cp (const Real T, Real * cp) {

  if (T <= 1000.0) {
    cp[0] = 8.24876732e+07 * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  } else {
    cp[0] = 8.24876732e+07 * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  }

  if (T <= 1000.0) {
    cp[1] = 4.12438366e+07 * (3.29812400e+00 + T * (8.24944200e-04 + T * (-8.14301500e-07 + T * (-9.47543400e-11 + 4.13487200e-13 * T))));
  } else {
    cp[1] = 4.12438366e+07 * (2.99142300e+00 + T * (7.00064400e-04 + T * (-5.63382900e-08 + T * (-9.23157800e-12 + 1.58275200e-15 * T))));
  }

  if (T <= 1000.0) {
    cp[2] = 5.19676363e+06 * (2.94642900e+00 + T * (-1.63816600e-03 + T * (2.42103200e-06 + T * (-1.60284300e-09 + 3.89069600e-13 * T))));
  } else {
    cp[2] = 5.19676363e+06 * (2.54206000e+00 + T * (-2.75506200e-05 + T * (-3.10280300e-09 + T * (4.55106700e-12 + -4.36805200e-16 * T))));
  }

  if (T <= 1000.0) {
    cp[3] = 4.88876881e+06 * (4.12530561e+00 + T * (-3.22544939e-03 + T * (6.52764691e-06 + T * (-5.79853643e-09 + 2.06237379e-12 * T))));
  } else {
    cp[3] = 4.88876881e+06 * (2.86472886e+00 + T * (1.05650448e-03 + T * (-2.59082758e-07 + T * (3.05218674e-11 + -1.33195876e-15 * T))));
  }

  if (T <= 1000.0) {
    cp[4] = 4.61523901e+06 * (3.38684200e+00 + T * (3.47498200e-03 + T * (-6.35469600e-06 + T * (6.96858100e-09 + -2.50658800e-12 * T))));
  } else {
    cp[4] = 4.61523901e+06 * (2.67214600e+00 + T * (3.05629300e-03 + T * (-8.73026000e-07 + T * (1.20099600e-10 + -6.39161800e-15 * T))));
  }

  if (T <= 1000.0) {
    cp[5] = 2.59838181e+06 * (3.21293600e+00 + T * (1.12748600e-03 + T * (-5.75615000e-07 + T * (1.31387700e-09 + -8.76855400e-13 * T))));
  } else {
    cp[5] = 2.59838181e+06 * (3.69757800e+00 + T * (6.13519700e-04 + T * (-1.25884200e-07 + T * (1.77528100e-11 + -1.13643500e-15 * T))));
  }

  if (T <= 1000.0) {
    cp[6] = 2.51903170e+06 * (4.30179801e+00 + T * (-4.74912051e-03 + T * (2.11582891e-05 + T * (-2.42763894e-08 + 9.29225124e-12 * T))));
  } else {
    cp[6] = 2.51903170e+06 * (4.01721090e+00 + T * (2.23982013e-03 + T * (-6.33658150e-07 + T * (1.14246370e-10 + -1.07908535e-14 * T))));
  }

  if (T <= 1000.0) {
    cp[7] = 2.44438441e+06 * (3.38875400e+00 + T * (6.56922600e-03 + T * (-1.48501300e-07 + T * (-4.62580600e-09 + 2.47151500e-12 * T))));
  } else {
    cp[7] = 2.44438441e+06 * (4.57316700e+00 + T * (4.33613600e-03 + T * (-1.47468900e-06 + T * (2.34890400e-10 + -1.43165400e-14 * T))));
  }

  if (T <= 1000.0) {
    cp[8] = 2.96804743e+06 * (3.29867700e+00 + T * (1.40824000e-03 + T * (-3.96322200e-06 + T * (5.64151500e-09 + -2.44485500e-12 * T))));
  } else {
    cp[8] = 2.96804743e+06 * (2.92664000e+00 + T * (1.48797700e-03 + T * (-5.68476100e-07 + T * (1.00970400e-10 + -6.75335100e-15 * T))));
  }

  if (T <= 1000.0) {
    cp[9] = 2.08133323e+06 * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  } else {
    cp[9] = 2.08133323e+06 * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  }

  if (T <= 1000.0) {
    cp[10] = 2.07727727e+07 * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  } else {
    cp[10] = 2.07727727e+07 * (2.50000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + T * (0.00000000e+00 + 0.00000000e+00 * T))));
  }

  if (T <= 1000.0) {
    cp[11] = 2.96834943e+06 * (3.26245200e+00 + T * (1.51194100e-03 + T * (-3.88175500e-06 + T * (5.58194400e-09 + -2.47495100e-12 * T))));
  } else {
    cp[11] = 2.96834943e+06 * (3.02507800e+00 + T * (1.44268900e-03 + T * (-5.63082800e-07 + T * (1.01858100e-10 + -6.91095200e-15 * T))));
  }

  if (T <= 1000.0) {
    cp[12] = 1.88923414e+06 * (2.27572500e+00 + T * (9.92207200e-03 + T * (-1.04091100e-05 + T * (6.86668700e-09 + -2.11728000e-12 * T))));
  } else {
    cp[12] = 1.88923414e+06 * (4.45362300e+00 + T * (3.14016900e-03 + T * (-1.27841100e-06 + T * (2.39399700e-10 + -1.66903300e-14 * T))));
  }

} // end eval_cp

