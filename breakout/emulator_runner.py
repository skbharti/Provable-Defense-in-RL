from multiprocessing import Process


class EmulatorRunner(Process):
    
    def __init__(self, id, emulators, variables, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.id = id
        self.emulators = emulators  ### each emulator runner process runs, these all emulators
        self.variables = variables
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()

    def _run(self):
        count = 0
        while True:  ### keep this process running until, the instruction none is sent   
            instruction = self.queue.get()
            # print('Signal recieved at emulator runner {0} : {1}'.format(self.id, instruction))
            if instruction is None:
                break
            for i, (emulator, action) in enumerate(zip(self.emulators, self.variables[-1])):
                new_s, reward, episode_over, _ = emulator.next(action)
                if episode_over:
                    self.variables[0][i] = emulator.get_initial_state()
                else:
                    self.variables[0][i] = new_s
                self.variables[1][i] = reward
                self.variables[2][i] = episode_over
            count += 1
            # print('Sending signal to central runner from emulator ', self.id)
            self.barrier.put('Emulator runner '+str(self.id)+' update completed!')



