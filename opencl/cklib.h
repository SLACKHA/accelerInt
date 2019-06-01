#ifndef __cklib_h
#define __cklib_h

#include <cl_macros.h>

#ifdef __cplusplus
extern "C"
{
#endif
/* H2CO */
/*enum {
    __ck_max_sp			= 14,
    __ck_max_rx			= 38,
    __ck_max_rev_rx		= 1,
    __ck_max_irrev_rx		= 1,
    __ck_max_thdbdy_rx		= 8,
    __ck_max_thdbdy_coefs	= 8*10,
    __ck_max_falloff_rx		= 3,
    __ck_max_falloff_params	=  8,

    __ck_max_strlen		= 16,
    __ck_max_th_terms		=  7,
    __ck_max_rx_order		=  3 // 3 reactants/products
};*/
/* GRI Mech v3 */
enum {
    __ck_max_sp			= 53,
    __ck_max_rx			= 325,
    __ck_max_rev_rx		= 1,
    __ck_max_irrev_rx		= 16,
    __ck_max_thdbdy_rx		= 41,
    __ck_max_thdbdy_coefs	= 41*10,
    __ck_max_falloff_rx		= 29,
    __ck_max_falloff_params	=  8,

    __ck_max_strlen		= 16,
    __ck_max_th_terms		=  7,
    __ck_max_rx_order		=  3 // 3 reactants/products
};
/* Ethylene */
/*enum {
    __ck_max_sp			= 111,
    __ck_max_rx			= 784,
    __ck_max_rev_rx		= 1,
    __ck_max_irrev_rx		= 2,
    __ck_max_thdbdy_rx		= 70,
    __ck_max_thdbdy_coefs	= (__ck_max_thdbdy_rx)*10,
    __ck_max_falloff_rx		= 62,
    __ck_max_falloff_params	=  8,

    __ck_max_strlen		= 16,
    __ck_max_th_terms		=  7,
    __ck_max_rx_order		=  3 // 3 reactants/products
};*/

#define __RU__	(8.314e7)
#define __PA__	(1.013250e6)

#define __small__	(1.e-300)
#define __big__		(1.e+300)
#define __exparg__	(690.775527898)

// ... binary switches
enum {
   __rx_flag_nil		= (0),		// Arrehius and equilibrium ... normal
   __rx_flag_irrev		= (1 << 1),	// Irreversible rxn
   __rx_flag_rparams		= (1 << 2),	// Reversible with explicit reverse rate params
   __rx_flag_thdbdy		= (1 << 3),	// 3-body efficiencies
   __rx_flag_falloff		= (1 << 4),	// Pressure dependencies ... and types
   __rx_flag_falloff_sri	= (1 << 5),
   __rx_flag_falloff_troe	= (1 << 6),
   __rx_flag_falloff_sri5	= (1 << 7),
   __rx_flag_falloff_troe4	= (1 << 7)
};

// ... Switch functions (for readability)
#define __is_enabled(__info__, __opt__) ( __info__ & __opt__ )
#define __enable(__info__, __opt__) ( __info__ |= __opt__ )
#define __disable(__info__, __opt__) ( __info__ &= ~(__opt__) )

typedef struct _ckdata_s
{
   // Species info

   int n_species;

   //char**	sp_name;
   double	sp_mwt[__ck_max_sp];//double*	sp_mwt;

   // Thermo info
   double	th_tmid[__ck_max_sp];//double*	th_tmid;
   //double	th_alo[__ck_max_sp * __ck_max_th_terms];//double*	th_alo;
   //double	th_ahi[__ck_max_sp * __ck_max_th_terms];//double*	th_ahi;
   double	th_alo[__ck_max_sp][__ck_max_th_terms];//double*	th_alo;
   double	th_ahi[__ck_max_sp][__ck_max_th_terms];//double*	th_ahi;

   // Reaction info

   int n_reactions;

   double	rx_A[__ck_max_rx];//double*	rx_A;
   double	rx_b[__ck_max_rx];
   double	rx_E[__ck_max_rx]; // normalized by R already ...

   //int		rx_nu[__ck_max_rx * __ck_max_rx_order*2];
   //int		rx_nuk[__ck_max_rx * __ck_max_rx_order*2];
   int		rx_nu[__ck_max_rx][__ck_max_rx_order*2];
   int		rx_nuk[__ck_max_rx][__ck_max_rx_order*2];
   int		rx_sumnu[__ck_max_rx];

   // Reversible reactions with explicit reversible parameters
   int n_reversible_reactions;

   int		rx_rev_idx[__ck_max_rev_rx];
   double	rx_rev_A[__ck_max_rev_rx];
   double	rx_rev_b[__ck_max_rev_rx];
   double	rx_rev_E[__ck_max_rev_rx];

   // Irreversible reactions ...
   int n_irreversible_reactions;

   int		rx_irrev_idx[__ck_max_irrev_rx];

   // 3rd-body efficiencies for pressure dependent reactions ...
   int n_thdbdy;

   int		rx_thdbdy_idx[__ck_max_thdbdy_rx];
   int		rx_thdbdy_offset[__ck_max_thdbdy_rx+1];
   int		rx_thdbdy_spidx[__ck_max_thdbdy_coefs];
   double	rx_thdbdy_alpha[__ck_max_thdbdy_coefs];

   // Fall-off reactions ...
   int n_falloff;

   int		rx_falloff_idx[__ck_max_falloff_rx];
   int		rx_falloff_spidx[__ck_max_falloff_rx];
   //double	rx_falloff_params[__ck_max_falloff_rx * __ck_max_falloff_params];
   double	rx_falloff_params[__ck_max_falloff_rx][__ck_max_falloff_params];

   int		rx_info[__ck_max_rx];

   // Internal Scratch data
   //int lenrwk_;
   //double*	rwk_;
}
ckdata_t;

ckdata_t* ck_create
          (int kk,
           char* sp_name[], double *sp_mwt,
           double *th_tmid, double *th_alo, double *th_ahi,
           int ii,
           double *rx_A, double *rx_b, double *rx_E,
           int *rx_nu, int *rx_nuk,
           int n_rev,
           int *rx_rev_idx, double *rx_rev_A, double *rx_rev_b, double *rx_rev_E,
           int n_irrev,
           int *rx_irrev_idx,
           int n_thdbdy,
           int *rx_thdbdy_idx, int *rx_thdbdy_offset, int *rx_thdbdy_spidx, double *rx_thdbdy_alpha,
           int n_falloff,
           int *rx_falloff_idx, int *rx_falloff_type, int *rx_falloff_spidx, double *rx_falloff_params);

size_t	ck_lenrwk (__ckdata_attr const ckdata_t *ck);
void    ck_destroy (ckdata_t **ck);

double  ckcpbs (const double T, __global const double y[], __ckdata_attr const ckdata_t *ck);
double  ckrhoy (const double p, const double T, __global const double y[], __ckdata_attr const ckdata_t *ck);
void    ckhms (const double T, __global double h[], __ckdata_attr const ckdata_t *ck);
void    ckytcp (const double p, const double T, __global const double y[], __global double c[], __ckdata_attr const ckdata_t *ck);
void    ckwyp (const double p, const double T, __global const double y[], __global double wdot[], __ckdata_attr const ckdata_t *ck, __global double rwk[]);
void    ckrhs (const double p, const double T, __global const double y[], __global double ydot[], __ckdata_attr const ckdata_t *ck, __global double rwk[]);
//double  ckrhoy (const double p, const double T, const double y[], const ckdata_t *ck);
//double  ckcpbs (const double T, const double y[], const ckdata_t *ck);
//double* ckwt (const ckdata_t *ck);
//double* ckrwk (const ckdata_t *ck);

int cklib_callback (int neq, double tcur, __global double *y, __global double *ydot, __private void *user_data);

//void *aligned_malloc (size_t size);

#ifdef __cplusplus
}
#endif

#endif
