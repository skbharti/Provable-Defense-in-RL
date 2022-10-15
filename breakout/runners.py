from builtins import breakpoint
import numpy as np
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float, c_double

### central class that manages all the child environment processes : each process(EmulatorRunner) \
#                                   handles (emulator_count/workers) number of workers in parallel.
class Runners(object):

    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8: c_uint}

    def __init__(self, EmulatorRunner, emulators, workers, variables):
        self.variables = [self._get_shared(var) for var in variables]   
        self.workers = workers
        self.queues = [Queue() for _ in range(workers)]         ### this queue list is used to send command to all worker processes
        self.barrier = Queue()                                  ### this central queue recieves the results from worker processes

        # split the data(32 emulators, variables) between 8(# of workers) creating 8# of (emulators, variables) tuples and create one emulator runner with each tuple
        zipped_data = zip(np.split(emulators, workers), zip(*[np.split(var, workers) for var in self.variables]))

        # create (here 8) number of emulator runners(each is a single python Processes) with 4 emulators in each
        self.runners = [EmulatorRunner(i, emulators, vars, self.queues[i], self.barrier) for i, (emulators, vars) in
                        enumerate(zipped_data)]

    def _get_shared(self, array):
        """
        Returns a RawArray backed numpy array that can be shared between processes; note that the 'variables' data location this process recieves is local to this process and  
        cannot be accessed from other processes normally, but by creating RawArray datatype a common memory block is obtained which can be shared between the processes
        :param array: the array to be shared
        :return: the RawArray backed numpy array
        """

        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]
        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        array_from_buffer = np.frombuffer(shared, dtype).reshape(shape)
        return array_from_buffer

    def start(self):
        for r in self.runners:
            r.start()

    def stop(self):
        ### send instruction None to the EmulatorRunner process to stop them
        for queue in self.queues:
            queue.put(None)

    def get_shared_variables(self):
        return self.variables

    def update_environments(self):
        for i, queue in enumerate(self.queues):
            # print('Sending signal to run emulator runner ',i)
            queue.put('Update emulator runner '+str(i))     ### the argument is the instruction sent to emulator runner

    def wait_updated(self):
        for wd in range(self.workers):
            result = self.barrier.get()
            # print("At Central Runner :  "+result)