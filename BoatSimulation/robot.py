class Robot:
    def __init__(self, simulation_timestep):
        self.simulation_timestep = simulation_timestep
        self.phi = []
        self.last_controller_time = 0.0
        self.last_torque = 0.0
            
        # TO EDIT:
        n = 1
        self.controller_frequency =  1.0 / (n * simulation_timestep)
        print(f"Controller frequency set to {self.controller_frequency} Hz")
        
        
    def get_phi(self):
        return self.phi[-1] if self.phi else None
    
    def get_torque(self):
        kP = 200;
        phi = self.get_phi()
        if phi is not None:
            return -kP * phi
        else:
            return 0.0


    # THESE FUNCTIONS ARE INTERFACING WITH THE SIMULATION --- DO NOT EDIT ---
    
    def external_sensor(self, phi):
        self.phi.append(phi) # allows external simulation to update "sensor" readings
        
    def controller(self, time):
        if(time - self.last_controller_time >= 1.0 / self.controller_frequency):
            self.last_controller_time = time
            self.last_torque = self.get_torque()
            return self.last_torque # is called during physics time steps to get the current control torque based on the latest sensor reading
        return self.last_torque # returns stale value if controller hasnt update