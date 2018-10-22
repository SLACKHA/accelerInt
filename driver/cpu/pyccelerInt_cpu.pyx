# distutils: language = c++

import cython
import numpy as np
cimport numpy as np
from libcpp cimport bool as bool_t  # noqa
from libcpp.string cimport string as string_t  # noqa
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
        EXPSolverOptions(double, double, bool_t, double, int, int) except +

cdef extern from "solver_interface.hpp" namespace "c_solvers":
    cdef cppclass Integrator:
        Integrator(int, int, SolverOptions&) except +
        ErrorCode integrate(const double, const double,
                            const double, double*) except +
        const double atol() except +
        const double rtol() except +
        const double neq() except +
        const double h_init() except +
        const double numThreads() except +
        const string_t& order() except +
        void getLog(const int, double*, double*) except +
        size_t numSteps() except +

    cdef cppclass SolverOptions:
        SolverOptions(double, double, bool_t, double) except +

    cdef cppclass IVP:
        IVP(size_t, bool_t) except +

    cdef unique_ptr[Integrator] init(IntegratorType, int, int,
                                     const IVP&,
                                     const SolverOptions&) except +

    cdef double integrate(Integrator&, const int, const double, const double,
                          const double, double * __restrict__,
                          const double * __restrict__)


cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[Integrator] move(unique_ptr[Integrator])

cdef class PyIntegrator:
    cdef unique_ptr[Integrator] integrator  # hold our integrator
    cdef PySolverOptions options # options, if not user specified
    cdef num # number of IVPs
    cdef neq # number of equations

    def __cinit__(self, IntegratorType itype, int neq, size_t numThreads,
                  PyIVP ivp, PySolverOptions options=None):
        if options is None:
            self.options = PySolverOptions(itype)
            options = self.options

        self.integrator = move(init(itype, neq, numThreads,
                                    deref(ivp.ivp.get()),
                                    deref(options.options.get())))
        self.num = -1
        self.neq = neq

    cpdef integrate(self, np.int32_t num, np.float64_t t_start,
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
            (:attr:`num`, :attr:`neq`, times.size)
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
        return times, np.reshape(phi, (self.num, self.neq, n_steps), order='F')

    def order(self):
        return 'F'


cdef class PySolverOptions:
    cdef unique_ptr[SolverOptions] options # hold our options

    def __cinit__(self, IntegratorType itype, double atol=1e-10, double rtol=1e-6,
                  bool_t logging=False, double h_init=1e-6,
                  int num_rational_approximants=10,
                  int max_krylov_subspace_dimension=-1):
        if itype in [IntegratorType.EXP4, IntegratorType.EXPRB43]:
            self.options.reset(
                new EXPSolverOptions(atol, rtol, logging, h_init,
                                     num_rational_approximants,
                                     max_krylov_subspace_dimension))
        else:
            self.options.reset(new SolverOptions(atol, rtol, logging, h_init))

cdef class PyIVP:
    cdef unique_ptr[IVP] ivp # hold IVP

    def __cinit__(self, size_t required_memory_size,
                  bool_t pass_entire_working_buffer=False):
        """
        Initialize the IVP object

        Parameters
        ----------
        required_memory_size: int
            The size of double-precision working memory (in bytes)
            required to evaluate the source terms and Jacobian

        pass_entire_working_buffer: bool [False]
            If True, pass the source term / Jacobian working memory buffer without
            offset to the user-functions.  If False, pass a pre-offset version --
            this option closely correponds to using stack-based memory, e.g.:

                const int work_size = 10;
                dydt()
                {
                    double rwk[work_size];
                    rwk[0] = 0;
                    ...
                }
        """

        self.ivp.reset(new IVP(required_memory_size, pass_entire_working_buffer))
