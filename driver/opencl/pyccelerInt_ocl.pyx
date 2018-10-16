# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=ascii

import cython
import numpy as np
cimport numpy as np
from libcpp cimport bool as bool_t
from libcpp.string cimport string as string_t
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from cython.operator cimport dereference as deref
from cpython.version cimport PY_MAJOR_VERSION

cdef unicode _bytes(s):
    py_byte_string = s.encode('UTF-8')
    cdef string_t c_string = py_byte_string
    return c_string

cdef extern from "solver_types.hpp" namespace "opencl_solvers":
    cpdef enum IntegratorType:
        ROS3,
        ROS4,
        RODAS3,
        RODAS4,
        RKF45

cdef extern from "error_codes.hpp" namespace "opencl_solvers":
    cpdef enum ErrorCode:
        SUCCESS,
        TOO_MUCH_WORK,
        TDIST_TOO_SMALL,
        MAX_STEPS_EXCEEDED

cdef extern from "solver_interface.hpp" namespace "opencl_solvers":
    cpdef enum DeviceType:
        CPU,
        GPU,
        ACCELERATOR,
        DEFAULT

    cdef cppclass IntegratorBase:
        IntegratorBase(int, size_t, const IVP&,
                       const SolverOptions&) except +
        const double atol() except +
        const double rtol() except +
        const double neq() except +
        void getLog(const int, double*, double*) except +
        size_t numSteps() except +
        string_t order() except+

    cdef cppclass IVP:
        IVP(const vector[string_t]&, size_t, size_t,
            const vector[string_t]&) except +

    cdef cppclass SolverOptions:
        SolverOptions(size_t, size_t, double, double,
                      bool_t, bool_t, string_t, string_t, DeviceType,
                      size_t, size_t) except +

        double atol() except+
        double rtol() except+
        bool_t logging() except+
        size_t vectorSize() except+
        size_t blockSize() except+
        string_t order() except+
        bool_t useQueue() except+
        string_t platform() except+
        DeviceType deviceType() except+
        size_t minIters() except+
        size_t maxIters() except+

    cdef unique_ptr[IntegratorBase] init(IntegratorType, int, int,
                                         const IVP&, const SolverOptions&) except +
    cdef unique_ptr[IntegratorBase] init(IntegratorType, int, int,
                                         const IVP&) except +

    cdef double integrate(IntegratorBase&, const int, const double, const double,
                          const double, double * __restrict__,
                          const double * __restrict__)

    cdef double integrate_varying(IntegratorBase&, const int, const double,
                          const double * __restrict__,
                          const double, double * __restrict__,
                          const double * __restrict__)

cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[IntegratorBase] move(unique_ptr[IntegratorBase])

cdef class PyIntegrator:
    cdef unique_ptr[IntegratorBase] integrator  # hold our integrator
    cdef num # number of IVPs
    cdef neq # number of equations

    def __cinit__(self, IntegratorType itype, int neq, size_t numThreads,
                  PyIVP ivp, PySolverOptions options=None):
        if options is not None:
            self.integrator = move(
                init(itype, neq, numThreads, deref(ivp.ivp.get()),
                     deref(options.options.get())))
        else:
            self.integrator = move(
                init(itype, neq, numThreads, deref(ivp.ivp.get())))
        self.num = -1
        self.neq = neq

    cpdef integrate(self, np.int32_t num, np.float64_t t_start,
                    np.float64_t t_end, np.ndarray[np.float64_t] phi_host,
                    np.ndarray[np.float64_t] param_host, np.float64_t step=-1):
        """
        Integrate :param:`num` IVPs, with varying start (:param:`t_start`) and
        end-times (:param:`t_end`)

        Parameters
        ----------
        num: int
            The number of IVPs to integrate
        t_start: double
            The integration start time
        t_end: double
            The integration end time
        phi_host: array of doubles
            The state vectors
        param: array of doubles
            The constant parameter
        step: double
            If supplied, use global integration time-steps of size :param:`step`.
            Useful for logging.
        """
        # store # of IVPs
        self.num = num
        return integrate(deref(self.integrator.get()), num, t_start,
                         t_end, step, &phi_host[0], &param_host[0])

    cpdef integrate_varying(
                    self, np.int32_t num, np.float64_t t_start,
                    np.ndarray[np.float64_t] t_end,
                    np.ndarray[np.float64_t] phi_host,
                    np.ndarray[np.float64_t] param_host, np.float64_t step=-1):
        """
        Integrate :param:`num` IVPs, with varying start (:param:`t_start`) and
        end-times (:param:`t_end`)

        Parameters
        ----------
        num: int
            The number of IVPs to integrate
        t_start: double
            The integration start time
        t_end: array of doubles
            The integration end times
        phi_host: array of doubles
            The state vectors
        param: array of doubles
            The constant parameter
        step: double
            If supplied, use global integration time-steps of size :param:`step`.
            Useful for logging.
        """

        # store # of IVPs
        self.num = num
        return integrate_varying(deref(self.integrator.get()), num, t_start,
                                 &t_end[0], step, &phi_host[0], &param_host[0])


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
        order = deref(self.integrator.get()).order()
        if order == 'C':
            phi_o = np.reshape(phi, (n_steps, self.num, self.neq), order=order)
            phi_o = np.swapaxes(phi_o, 0, 1).swapaxes(1, 2)
        else:
            phi_o = np.reshape(phi, (self.num, self.neq, n_steps), order=order)
        return times, phi_o


cdef class PySolverOptions:
    cdef unique_ptr[SolverOptions] options # hold our options

    def __cinit__(self, IntegratorType itype, size_t vectorSize=1,
                  size_t blockSize=1, double atol=1e-10,
                  double rtol=1e-6, bool_t logging=False, bool_t use_queue=True,
                  string_t order="C", string_t platform="",
                  DeviceType deviceType = DeviceType.DEFAULT,  size_t minIters=1,
                  size_t maxIters = 1000):

        cdef string_t c_order = _bytes(order)
        self.options.reset(new SolverOptions(
            vectorSize, blockSize,
            atol, rtol, logging, use_queue,
            c_order, platform, deviceType,
            minIters, maxIters))

    cpdef atol(self):
        return deref(self.options.get()).atol()

    cpdef rtol(self):
        return deref(self.options.get()).rtol()

    cpdef logging(self):
        return deref(self.options.get()).logging()

    cpdef vector_size(self):
        return deref(self.options.get()).vectorSize()

    cpdef block_size(self):
        return deref(self.options.get()).blockSize()

    cpdef order(self):
        return deref(self.options.get()).order()

    cpdef use_queue(self):
        return deref(self.options.get()).useQueue()

    cpdef platform(self):
        return deref(self.options.get()).platform()

    cpdef device_type(self):
        return deref(self.options.get()).deviceType()

    cpdef min_iters(self):
        return deref(self.options.get()).minIters()

    cpdef max_iters(self):
        return deref(self.options.get()).maxIters()


cdef class PyIVP:
    cdef unique_ptr[IVP] ivp # hold our ivp implementation
    cdef vector[string_t] source
    cdef vector[string_t] includes
    cdef int mem
    cdef int imem

    def __cinit__(self, kernel_source, int required_memory,
                  int required_int_memory=0, include_paths=[]):
        """
        Create an IVP implementation object, from:

        Parameters
        ----------
        kernel_source: iterable of str
            The paths to the kernel source files to use
        required_memory: int
            The amount of double-precision floating-point memory (in bytes)
            required per-IVP.  Note: this should _not_ include any vectorization
            considerations.
        required_int_memory: int
            The amount of integer memory (measured in bytes)
            required per-IVP.  Note: this should _not_ include any vectorization
            considerations.
        include_paths: iterable of str
            The necessary include paths for this kernel
        """

        for x in kernel_source:
            assert isinstance(x, basestring), "Kernel path ({}) not string!".format(
                x)
            self.source.push_back(_bytes(x))
        for x in include_paths:
            assert isinstance(x, basestring), "Include path ({}) not string!".format(
                x)
            self.includes.push_back(_bytes(x))

        self.mem = required_memory
        self.imem = required_int_memory

        self.ivp.reset(new IVP(self.source, self.mem, self.imem, self.includes))
