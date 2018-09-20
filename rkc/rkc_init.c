/**
 * \file
 * \brief Implementation of the necessary initialization for the RKC solver
 *
 * \author Nicholas Curtis
 * \date 08/12/2017
 *
 */

#ifdef GENERATE_DOCS
namespace rkc {
#endif

 void initialize_solver() {
 }

/*!
   \fn char* solver_name()
   \brief Returns a descriptive solver name
*/
 const char* solver_name() {
    const char* name = "rkc-int";
    return name;
 }

 void cleanup_solver() {
    //nothing to do
 }

 void init_solver_log() {

 }

 void solver_log() {

 }

#ifdef GENERATE_DOCS
}
#endif