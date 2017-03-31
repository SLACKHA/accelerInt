/**
 * \file
 * \brief More cruft needed for pyJac, slated to be removed in a future release
 *
 * This provides simple (empty) definitions of the mask functions so that they're not multiply defined
 */

#ifdef GENERATE_DOCS
//put this in the van der Pol namespace for documentation
namespace van_der_pol {
#endif

/**
 * \brief Not needed for van der Pol
 *
 *  In pyJac, these are used to transform the input/output vectors to deal with moving the
 *         last species mass fraction
 */
void apply_mask(double* y_host) {}
/**
 * \brief Not needed for van der Pol
 *
 *  In pyJac, these are used to transform the input/output vectors to deal with moving the
 *         last species mass fraction
 */
void apply_reverse_mask(double* y_host) {}

#ifdef GENERATE_DOCS
}
#endif