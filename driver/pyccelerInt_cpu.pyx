# distutils: language = c++

import cython
import numpy as np
cimport numpy as np
from libcpp cimport bool as bool_t  # noqa
from libcpp.memory cimport unique_ptr
from cython.operator cimport dereference as deref

cdef extern from "solver_types.hpp" namespace "c_solvers":
    cpdef enum IntegratorType:
        RADAU_II_A,
        EXP4,
        EXPRB43,
        RK_78,
        RKC,
        CVODES

cdef extern from "error_codes.hpp" namespace "c_solvers":
    cpdef enum ErrorCode:
        SUCCESS,
        MAX_CONSECUTIVE_ERRORS_EXCEEDED,
        MAX_STEPS_EXCEEDED,
        H_PLUS_T_EQUALS_H,
        MAX_NEWTON_ITER_EXCEEDED

cdef extern from "exp_solver.hpp" namespace "c_solvers":
    cdef cppclass EXPSolverOptions:
        EXPSolverOptions(double, double, bool, int, int) except +

cdef extern from "solver_interface.hpp" namespace "c_solvers":
    cdef cppclass Integrator:
        Integrator(int, int, double, double) except +
        ErrorCode integrate(const double, const double,
                            const double, double*) except +
        const double atol() except +
        const double rtol() except +
        const double neq() except +
        const double numThreads() except +
        void getLog(const int, double*, double*) except +
        size_t numSteps() except +

    cdef cppclass SolverOptions:
        SolverOptions(double, double, bool) except +

    cdef unique_ptr[Integrator] init(IntegratorType, int, int,
                                     const SolverOptions&) except +
    cdef unique_ptr[Integrator] init(IntegratorType, int, int) except +

    cdef double integrate(Integrator&, const int, const double, const double,
                          const double, double * __restrict__,
                          const double * __restrict__)

cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[Integrator] move(unique_ptr[Integrator])

cdef class PyIntegrator:
    cdef unique_ptr[Integrator] integrator  # hold our integrator
    cdef num # number of IVPs
    cdef neq # number of equations

    def __cinit__(self, IntegratorType itype, int neq, size_t numThreads,
                  PySolverOptions options=None):
        if options is not None:
            self.integrator = move(init(itype, neq, numThreads,
                                        deref(options.options.get())))
        else:
            self.integrator = move(init(itype, neq, numThreads))

    def __init__(self, itype, neq, numThreads, options):
        self.num = -1
        self.neq = neq

    def integrate(self, np.int32_t num, np.float64_t t_start,
                  np.float64_t t_end, np.ndarray[np.float64_t] y_host,
                  np.ndarray[np.float64_t] var_host, np.float64_t step=-1):

        # store # of IVPs
        self.num = num
        return integrate(deref(self.integrator.get()), num, t_start,
                         t_end, step, &y_host[0], &var_host[0])

    def state(self):
        """
        Returns
        -------
        times: np.ndarray
            The array of times that this integrator has reached
        state: np.ndarray
            The state vectors at each time, shape is
            (times.size, :attr:`num`, :attr:`neq`)
        """
        assert self.num > 0 and self.neq > 0
        n_steps = deref(self.integrator.get()).numSteps()
        cdef np.ndarray[np.float64_t, ndim=1] phi = np.zeros(
            self.num * self.neq * n_steps, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] times = np.zeros(
            n_steps, dtype=np.float64)
        # get phi
        deref(self.integrator.get()).getLog(self.num, &times[0], &phi[0])
        # and reshape
        return times, np.reshape(phi, (n_steps, self.num, self.neq), order='F')


cdef class PySolverOptions:
    cdef unique_ptr[SolverOptions] options # hold our options

    def __cinit__(self, IntegratorType itype, double atol=1e-10, double rtol=1e-6,
                  bool_t logging=False, int num_rational_approximants=10,
                  int max_krylov_subspace_dimension=-1):
        if itype in [IntegratorType.EXP4, IntegratorType.EXPRB43]:
            self.options.reset(
                new EXPSolverOptions(atol, rtol, logging,
                                     num_rational_approximants,
                                     max_krylov_subspace_dimension))
        else:
            self.options.reset(new SolverOptions(atol, rtol, logging))
