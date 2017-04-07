#Copyright 2014 MathWorks, Inc.

"""
FutureResult: The class name of a future result returned by the MATLAB Engine.

An instance of FutureResult is returned from each asynchronous invocation of a
MATLAB statement or function.  The future result serves as a placeholder of the
actual result, so the future result can be returned immediately.  MATLAB puts
the actual result into the placeholder when the MATLAB function finishes its
evaluation.  The future result can be used to interrupt the execution, check the
completion status, and get the result of a MATLAB statement.

"""

from matlab.engine import pythonengine
from matlab.engine import RejectedExecutionError
from matlab.engine import TimeoutError
import time
import weakref

try:
    long
except NameError:
    long = int

class FutureResult():
    """
    A FutureResult object is used to hold the future result of a MATLAB 
    statement.  The FutureResult object should be only created by MatlabEngine 
    after submitting a MATLAB command for evaluation.
    """
    
    def __init__(self, eng, handle, nout, stdout, stderr):
        self._engine = weakref.ref(eng)
        self._future = handle
        self._nargout = nout
        self._out = stdout
        self._err = stderr
        self._retrieved = False
        self._result = None
        
    def result(self, timeout=None):
        """
        Get the result of a MATLAB statement.

        Parameter
            timeout: int
                    Number of seconds to wait before returning.  By default,
            this function will wait until the result is generated.

        Returns
            The result of MATLAB statement.  A tuple is returned if multiple
            outputs are returned.

        Raises
            SyntaxError - if there is an error in the MATLAB statement.
            InterruptedError - if the task is interrupted.
            CancelledError - if the evaluation of MATLAB function is cancelled already.
            TimeoutError - if this method fails to get the result in timeout seconds.
            MatlabExecutionError - if the MATLAB statement fails in execution.
            TypeError - if the data type of return value is not supported.
            RejectedExecutionError  - an error occurs if the engine is terminated.
        """ 
        self.__validate_engine()
        
        if timeout is not None:
            if not isinstance(timeout, (int, long, float)):
                raise TypeError("timeout must be a number, not a %s" %
                                type(timeout).__name__)
            
            if timeout < 0:
                raise TypeError("timeout cannot be a negative number")
        
        if self._retrieved:
            return self._result

        """
        Following code is used to poll the Ctrl+C every second from keyboard in
        order to cancel a MATLAB function.
        """
        time_slice = 1
        try:
            if timeout is None:
                result_ready = self.done()
                while not result_ready:
                    result_ready = pythonengine.waitForFEval(self._future,
                                                             time_slice)
            else:
                result_ready = self.done()
                current_time = time.time()
                sleep_until = current_time + timeout
                while (not result_ready) and (current_time < sleep_until):
                    if (sleep_until - current_time) >= time_slice:
                        result_ready = pythonengine.waitForFEval(self._future,
                                                                 time_slice)
                    else:
                        time_to_sleep = sleep_until - current_time
                        result_ready = pythonengine.waitForFEval(self._future,
                                                                 time_to_sleep)
                    current_time = time.time()

                if not result_ready:
                    raise TimeoutError(
                        'timeout from the execution of the MATLAB function')

            self._result = pythonengine.getFEvalResult(
                self._future,self._nargout, None, out=self._out, err=self._err)
            self._retrieved = True
            return self._result

        except KeyboardInterrupt:
            self.cancel()
            print("the MATLAB function is cancelled")
        except:
            raise
    
    def cancel(self):
        """
        Cancel the execution of an evaluation of a MATLAB statement.
    
        Returns 
            bool - True if the corresponding MATLAB statement can be cancelled;
            False otherwise.

        Raises 
            RejectedExecutionError  - an error occurs if the engine is terminated.
        """
        self.__validate_engine()
        return pythonengine.cancelFEval(self._future)
    
    def cancelled(self):
        """
        Obtain the cancellation status of the asynchronous execution of a MATLAB
        command.
    
        Returns 
            bool - True if the execution is cancelled; False otherwise.

        Raises 
            RejectedExecutionError  - an error occurs if the engine is terminated.
        """
        self.__validate_engine()
        return pythonengine.isCancelledFEval(self._future)

    def done(self):
        """
        Obtain the completion status of the asynchronous invocation of a MATLAB
        command.

        Returns 
            bool - True if the execution is finished; False otherwise.  It
            returns True even if there is an error generated from the MATLAB
            statement or it is cancelled.

        Raises 
            RejectedExecutionError - an error occurs if the engine is terminated.
        """
        self.__validate_engine()
        return pythonengine.isDoneFEval(self._future)
    
    def __del__(self):
        if self._future is not None:
            pythonengine.destroyFEvalResult(self._future)
            self._future = None
            self._result = None
            
    def __validate_engine(self):
        if self._engine() is None or not self._engine()._check_matlab():
            raise RejectedExecutionError("MATLAB has already terminated")  