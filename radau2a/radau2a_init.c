/**
 * \file
 * \brief Implementation of the necessary initialization for the RadauII-A solver
 *
 * \author Nicholas Curtis
 * \date 03/09/2015
 *
 */

#ifdef GENERATE_DOCS
namespace radau2a {
#endif

 void initialize_solver() {
 }

/*!
   \fn char* solver_name()
   \brief Returns a descriptive solver name
*/
 const char* solver_name() {
 	const char* name = "radau2a-int";
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