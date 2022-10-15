import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import colors as colors

from numpy import linalg as LA

from scipy import linalg as sciLA

from scipy.sparse import csr_matrix

import pandas as pd




def Hill(x, K, n):
    # The Hill function
    return x**n/(x**n+K**n)   

def Heavi(x,K):
    # The Heaviside function
    if x>=K:
        return 1.0
    else:
        return 0.0


def delta(x, h):
    # The delta function for probability mass distribution
    # This form of discrete delta is widely used in the immersed boundary method.
    # Note that the lattice size h should be supplied, which is not necessary when used with LBM.
    # Note that this is only the 1D delta. For 2D case, delta*delta should be used.
    abs = np.abs
    sqrt = np.sqrt 
    
    x0 = abs(x/h)
    
    if x0<=1:
        return (3 - 2*x0 + sqrt(1+4*x0-4*x0*x0))/8/h
    elif x0<=2:
        return (5 - 2*x0 - sqrt(-7+12*x0-4*x0*x0))/8/h
    else:
        return 0    

class Pulse():
    # Pulse (actin wave) class
    # The basic parameters are pulse rate and amplitude.
    # This is the pulse for a single equation. For the 2D case, currently both equations use the same pulse term. So the effective pulse rate is 2*rate 
    def __init__(self, pars=[1,1]):
        amp, rate = pars
        self.amp = amp
        self.rate = rate
        
class Neurite():
    # Neurite class
    def __init__(self, pars):
        beta, Kg, ng, r = pars # Note that the growth rate is denoted by ``beta'', which is ``g'' in the paper.
        self.beta = beta
        self.Kg = Kg
        self.ng = ng
        self.r = r
        self.r0 = r # base retraction rate
        
    def A(self, x):
        # The RHS of the deterministic equation
        # This function is not used anymore.
        return self.beta*(x**self.n)/(x**self.n+self.K**self.n) - self.r*x

    
    def plot_growth_retraction_terms(self):
        # Plot beta*(x**n)/(x**n+K**n) and r*x separately
        # This is used to show whether there is bistability in the deterministic part of the equation for a single neurite.
        beta = self.beta
        Kg = self.Kg
        ng = self.ng
        r = self.r
        
        x_list = np.linspace(0, 2*Kg, 101)
        growth_list = []
        rL_list = []
        for x in x_list:
            growth_list.append(beta*Hill(x, Kg, ng))
            rL_list.append(r*x)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(x_list, growth_list, label='Growth')
        ax.plot(x_list, rL_list, label='Retraction')
        ax.legend(loc='best')
        ax.set_xlabel('x')
        ax.set_title(r"$\beta={},  K_g={:.2f},  n_g={},  r={}$".format(beta, Kg, ng, r))

        
class Cell(): 
    # The lattice cell class
    def __init__(self, pars):
        label, pos, dx, pos_min, pos_max = pars # Extract input parameters
        self.label = label # The label of a cell is a tuple of integers. The first cell is labeled as [0,0,...] Does it depend on the dimension????
        self.pos = pos # The position of a cell is its center location
        self.dx = dx # The increments in all directions
        
        self.pos_min = pos_min # min position of the cell in all directions
        self.pos_max = pos_max # max position of the cell in all directions
        
        self.neighbors = [] # Neighboring cells, may not be used
        
        self.d_to_target = 0 # Distance to a target position, used to spread prob mass
        
        self.prob_spread = 0 # The probability mass to be spread, used to spread prob mass

        # The number of end positions in this cell. 
        # Used when calculating Q
        self.num_endpos = 0 
        
class Probs():
    # Probability class, used to store probability distribution in various forms
    def __init__(self, dim):
        self.dim = dim # Dimension of the system, equal to #neurites.
        self.N = 0 # To store the total number of cells
        self.cells = [] # Cell list
        
        # The following three lists store prob distributions
        # These are vectors. The cells are ordered in a certain way in order to get these vectors.
        # To plot the distribution for the 2D case, the distribution must be transformed into a matrix.
        self.P0 = [] # Probability list for the previous time step, may not be used
        self.P1 = [] # Probability list
        self.Ps = [] # Equilibirum distribution (the eigenvector corresponding to eigenvalue 1)
        
        self.Q = [] # Transition matrix
        self.P1_mat = [] # Matrix for the distribution at some time. This is for plotting in 2D case.
        self.P0_mat = [] # Matrix for the initial distribution 
        self.Ps_mat = [] # Matrix for the stationary distribution. This is for plotting in 2D case.
        self.map_labelpair_indexP = {} # The labels are those of the cells. The index refers to their location in self.P.
        self.map_indexP_labelpair = {} # The inverse map of the previous one
        
        self.Q_evals = [] # The eigenvalues of Q. They are used in finding stationary distributions and testing recurrency
        self.Q_evects = [] # The eigenvectors of Q. Each column corresponds to an eigenvector.
        self.Q_evects_inv = [] # The inverse of Q_evects, used in diagonalization.
        self.Q_inf = [] # Q^{inf}: the stationary transition matrix
        self.Q_n = [] # Q^n
        self.Q_2expn = [] # Q^{2n} # This is used to get the final Q fast.
        self.Q_absorb = [] # Modified Q according to the absorbing region, used to calculate FPT
        

        
        
        # Sparse version of the matrices and vectors
        # Used for fast multiplication
        self.Q_sparse = [] 
        self.P1_sparse = []
        self.P0_sparse = []
        
        
        # For Markov chain analysis
        self.Q_list = []
        
        
        # Transpose of Q
        self.Q_T = []
        # eps-committor
        self.q_eps = []
        # eps-committor field
        self.q_eps_field = []
        # Mean dwell time field, proportional to eps-committor field
        self.mean_dwell_time = []
        
        
        
        self.dt = 0 # Time step
        self.t = 0 # Current time
        
        self.neurites = [] # Neurite list
        self.pulses = [] # Pulse list
        
        
        self.xmin = [] # lower boundary for all directions
        self.xmax = [] # upper boundary for all directions
        self.dx = [] # increments for all directions
        
        self.X = [] # Position mesh for 2D colormaps of Q and P_mat
        self.Y = [] # Position mesh for 2D colormaps of Q and P_mat
        
        self.num_cells = [] # Number of cells in all directions. Has this ever been used?????
        
       
        
        # Index range for cells in all directions
        # Note that if there are n cells in a direction, the index ranges from 0 to n-1
        self.index_max = []
        # The number of cells in two directions
        self.n_cells = []
        
      
    def set_dt(self, dt):
        # Simply assign the value of dt
        self.dt = dt
        
        
    def set_range(self, xmin, xmax, dx):
        # Set the range for all directions
        # Note that these boundarys are vectors, which means the boudaries of all directions
        # Even for 1D case, the pars should be supplied as lists.
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        # Calculate the index range for all directions
        # self.index_max = [max index of the 1st direction, max index of the 2nd direction, ...]
        # Note that if there are n cells in a direction, the index ranges from 0 to n-1
        # The total length of all cells in a direction may be longer than xmax-xmin. 
        # For the last cell, maybe only part of it is within [xmin, xmax].(Here, [] means interval)
        for i in range(self.dim):
            self.index_max.append(int(np.ceil((self.xmax[i]-self.xmin[i])/self.dx[i]))-1)
            # The number of cells in a direction is 1 larger than the max label.
            self.n_cells.append(self.index_max[-1]+1)
            
            
    def create_cells_n(self):
        # Create cells 
        # Use alias to simplify codes
        xmin = self.xmin
        #xmax = self.xmax
        dx =self.dx
        index_max = self.index_max
        
        # Create cells for 1D case (a single neurite)
        if self.dim == 1:
            for n in range(index_max[0]+1): # Note that index_max stores the maximum indices of cells in all directions. In order for n to reach index_max, we must use index_max[i]+1, not index_max.
                label = (n,) # Generate the label for a cell. Pay attention to the format.
                pos = [ xmin[0] + dx[0]/2 + n*dx[0],] # Generate the position for a cell. Pay attention to the format.
                pos_min = [xmin[0] + n*dx[0],]
                pos_max = [xmin[0] + dx[0] + n*dx[0],]
                
                cell = Cell([label, pos, dx.copy()])  # Generate a cell
                
                self.cells.append(cell) # Put the cell in the cell list
                self.map_labelpair_indexP[label] = self.N # Add the label-index relation to the relation dictionary.
                self.map_indexP_labelpair[self.N] = label # Add the index-label relation to the relation dictionary.
                self.N += 1 # Increase the total number of cells by 1
        
        # Create cells for 2D case (a single neurite)
        # This is similar to the 1D case. Everything done in the 1D case is done twice here.
        if self.dim == 2:
            for n1 in range(self.index_max[1] + 1): 
                for n0 in range(self.index_max[0] + 1): 
                    # Generate the cell label pair
                    label = (n0, n1)
                    # Calculate the center position
                    pos = []
                    pos.append(xmin[0] + dx[0]/2 + n0*dx[0])
                    pos.append(xmin[1] + dx[1]/2 + n1*dx[1])
                    # Calculate the lower boundary in each direction
                    pos_min = []
                    pos_min.append(xmin[0] + n0*dx[0])
                    pos_min.append(xmin[1] + n1*dx[1])
                    # Calculate the upper boundary in each direction
                    pos_max = []
                    pos_max.append(xmin[0] + dx[0] + n0*dx[0])
                    pos_max.append(xmin[1] + dx[1] + n1*dx[1])
                    # Generate a cell
                    cell = Cell([label, pos, dx.copy(), pos_min, pos_max])
                    # Add the cell to the cell list
                    self.cells.append(cell)
                    # Create maps between cell indices and label pairs
                    self.map_labelpair_indexP[label] = self.N
                    self.map_indexP_labelpair[self.N] = label
                    self.N += 1   
                    
                    
    def init_P_Q(self):
        # Initialize vectors P0, P1 and Q
        N = self.N
        self.P0 = np.zeros(N)
        self.P1 = np.zeros(N)
        self.Q = np.zeros((N,N))
        
        if self.dim == 2:
            self.P1_mat = np.zeros((self.index_max[0]+1, self.index_max[1]+1))
            self.Ps_mat = np.zeros((self.index_max[0]+1, self.index_max[1]+1))
            self.P0_mat = np.zeros((self.index_max[0]+1, self.index_max[1]+1))
 
        
    def set_P(self, pos_init):
        # Initial distribution is a delta function at 0 or (0,0)
        # Use the distribute function to spread the delta 
        
        # Reset P1
        self.P1 = np.zeros(self.N)
        # Reset P0
        self.P0 = np.zeros(self.N)
        
        
        
        # Find the cells to be spread and the corresponding prob mass   
        
        cell_list_distributed = self.distribute(pos=pos_init, prob_total=1)
        
        # pos_init should be different for 1D and 2D case, see below
        #if self.dim == 1: 
        #    cell_list_distributed = self.distribute(pos=[0,], prob_total=1)
        #elif self.dim ==2:
        #    cell_list_distributed = self.distribute(pos=[0,0], prob_total=1)
        
        
        # Put the spread prob mass at the position in P0 and P1
        for cell_distributed in cell_list_distributed:
            indexP = self.map_labelpair_indexP[cell_distributed.label]
            self.P1[indexP] = cell_distributed.prob_spread
            self.P0[indexP] = cell_distributed.prob_spread
        
        # Reset prob_spread for each cell
        for cell_distributed in cell_list_distributed:
            cell_distributed.prob_spread = 0
    
    def set_P_not_spread(self, pos_init):
        # Set P according to initial position
        # The cell containing the initial position acquires prob of 1
        # Used only for dim==2
        # Reset P1
        self.P1 = np.zeros(self.N)
        # Reset P0
        self.P0 = np.zeros(self.N)
        
        # Find the cell that contains pos_init
        cell_init = self.pos_to_cell_dim2(pos_init)
        
        indexP = self.map_labelpair_indexP[cell_init.label]
        self.P1[indexP] = 1
        self.P0[indexP] = 1
    
    def create_neurites_n(self, pars_all_neurites):
        # Create neurites 
        # pars_all_neurites=[[pars of the first neurite],[pars of the second neurites]]
        for pars in pars_all_neurites:
            neurite = Neurite(pars)
            self.neurites.append(neurite)
        
    def create_pulse(self, pars):
        # Create pulse
        # Currently, only one pulse object is needed.
        pulse = Pulse(pars)
        self.pulses.append(pulse)
        
    
    def cal_Q(self, num_tau_invervals):
        # Calculate Q for the 2D case
        dt = self.dt 
        dtau = dt/num_tau_invervals # dtau further divides dt. It is used to deal with uniform distribution of jumps within dt.
        # Get the base pulse rate
        pulse_rate_base = self.pulses[0].rate # Currently, there is only a single pulse object.
        # Get the base pulse amp
        pulse_amp_base = self.pulses[0].amp
        pulse_amp = self.pulses[0].amp
        for cell in self.cells:
            # Get the starting position (the center of the current cell)
            pos_start = cell.pos
            # Update the state-dependent pulse rate and amp
            # This is the crucial part. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Unquote the codes to use
            
            # The total length, used to change the rate or/and amplitude
            sum_length = cell.pos[0] + cell.pos[1]
            
           
            # Combination of retraction rate change and pulse rate change
            mu = 0.1
            pulse_rate = pulse_rate_base/(1+mu*(sum_length))
            pulse_amp = pulse_amp_base
            
           
            
            # Finite pool of proteins
            mu = 0.0
            pulse_rate = pulse_rate_base/(1+mu*(sum_length))
            Lt_half = 10
            pulse_amp = pulse_amp_base * Lt_half**2 / (Lt_half**2+(sum_length)**2)
            
            
            # Change the rate
            mu = 1
            pulse_rate = pulse_rate_base/(1+mu*(sum_length))
            pulse_amp = pulse_amp_base
            
            
            # No change, base rate and amp
            mu = 0.0
            pulse_rate = pulse_rate_base/(1+mu*(sum_length))
            pulse_amp = pulse_amp_base
           
            # Deterministic part
            # Calculate the end position
            pos_end = self.cal_pos_deterministic_n(ICs=pos_start, duration=dt)
            # Calculate the total prob to be spread
            # Note that the equation for each neurite has a pulse term, so the total pulse rate is 2*pulse_rate
            #prob_deterministic = 1 - 2 * pulse_rate * dt
            prob_deterministic = 1 - pulse_rate * dt
            #prob_deterministic = 1 ########Test the deterministic part!!!!! Deleted later!!!!!!
            # Find the cells to be spread
            cell_list_distributed = self.distribute(pos_end, prob_deterministic)
            # Find the second index of Q
            # This should be the index corresponding to the current cell.
            # Meaning of Qij is the prob of reaching i from j.?????
            index_Q_2 = self.map_labelpair_indexP[cell.label]
            # Spread the probs to Q
            self.distribute_Q(index_Q_2, cell_list_distributed)
            
           
            
            # Jump part
            # Assume that the pulse arrives in the middle of each dtau
            # The prob mass to be spread
            prob_jump = pulse_rate/self.dim * dtau # In fact, this is pulse_rate / self.dim * dt * dtau / dt
            for tau in np.arange(dtau/2, dt, dtau):
                # The jump in the first neurite length
                amp_jump = [pulse_amp, 0]
                # Calculate the final postion after dt
                pos_end = self.cal_pos_jump_n(ICs=pos_start, duration=dt, t_jump=tau, amp_jump=amp_jump)
                # Get the cell list where the prob mass to be spread to
                cell_list_distributed = self.distribute(pos_end, prob_jump)
                # Distribute the prob mass to these cells
                self.distribute_Q(index_Q_2, cell_list_distributed)
                
                # The jump in the second neurite length   
                amp_jump = [0, pulse_amp]
                pos_end = self.cal_pos_jump_n(ICs=pos_start, duration=dt, t_jump=tau, amp_jump=amp_jump)
                cell_list_distributed = self.distribute(pos_end, prob_jump)
                self.distribute_Q(index_Q_2, cell_list_distributed)
               
            
    def cal_Q_random(self, num_pt, num_tau_invervals):          
        # Calculating Q by randomly sampling trajectories
        # This is used only in the 2D case
        # Calculate Q for the 2D case
        dt = self.dt 
        dtau = dt/num_tau_invervals # dtau further divides dt. It is used to deal with uniform distribution of jumps within dt.
        # Get the base pulse rate
        pulse_rate_base = self.pulses[0].rate # Currently, there is only a single pulse object.
        # Get the base pulse amp
        pulse_amp_base = self.pulses[0].amp
                
               
        for cell in self.cells:
            print("Working on cell", cell.label)
            # Initiate cell list to spread probability mass
            cell_list_distributed = []
            # Sample points uniformly in the cell
            pts = np.random.uniform(cell.pos_min, cell.pos_max, (num_pt, self.dim))
            
            # Find the end positions
            # calculate Qij
            
            # Sum of the lengths
            # Assume that the lengths are the same as the center positions of the cell
            sum_length = cell.pos[0] + cell.pos[1]
            # Update the state-dependent pulse rate and amp
            # This is the crucial part. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Unquote the codes to use
                      
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            
            # Finite pool of proteins
            #mu = 0.0
            #pulse_rate = pulse_rate_base/(1+mu*(sum_length))
            #Lt_half = 10
            #pulse_amp = pulse_amp_base * Lt_half**2 / (Lt_half**2+(sum_length)**2)
            
            
            
            # Change the amp
            phi = 0.5
            pulse_rate = pulse_rate_base
            pulse_amp = pulse_amp_base/(1+phi*(sum_length))
            
            # Change the rate
            mu = 0.5
            pulse_rate = pulse_rate_base/(1+mu*(sum_length))
            pulse_amp = pulse_amp_base
            
            # No change, base rate and amp
            pulse_rate = pulse_rate_base
            pulse_amp = pulse_amp_base
            
            
            
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # Calculate the total prob to be spread for the deterministic motion
            # The total pulse rate is pulse_rate
            # Each neurite's pulse rate is pulse_rate/self.dim
            prob_deterministic = 1 - pulse_rate * dt
            prob_jump = pulse_rate/self.dim * dtau            
            
            
            for pos_start in pts:
                
                ####### Deterministic part
                # Calculate the end position
                pos_end = self.cal_pos_deterministic_n(ICs=pos_start, duration=dt)
                
                # Find the cell where the end position lies
                cell_end = self.pos_to_cell_dim2(pos_end)
                
                
                # Add the prob mass to this cell
                cell_end.prob_spread += prob_deterministic/num_pt
               
                
                # Add this cell to the cell list, if it is not there
                if not (cell_end in cell_list_distributed):
                    cell_list_distributed.append(cell_end)
                
                
                
                ####### Jump part
                # The prob mass to be spread
                # For each neurite, the pulse rate is pulse_rate/2
                prob_jump = pulse_rate/2 * dtau
                # Assume that the pulse arrives in the middle of each dtau
                for tau in np.arange(dtau/2, dt, dtau):
                    # The jump in the first neurite length
                    amp_jump = [pulse_amp, 0]
                    # Calculate the final postion after dt
                    pos_end = self.cal_pos_jump_n(ICs=pos_start, duration=dt, t_jump=tau, amp_jump=amp_jump)
                    # Find the cell where the end position lies
                    cell_end = self.pos_to_cell_dim2(pos_end)
                    # Add the prob mass to this cell
                    cell_end.prob_spread += prob_jump/num_pt
                    # Add this cell to the cell list, if it is not there
                    if not (cell_end in cell_list_distributed):
                        cell_list_distributed.append(cell_end)
                    
                    
                    
                    # The jump in the second neurite length   
                    amp_jump = [0, pulse_amp]
                    pos_end = self.cal_pos_jump_n(ICs=pos_start, duration=dt, t_jump=tau, amp_jump=amp_jump)
                    # Find the cell where the end position lies
                    cell_end = self.pos_to_cell_dim2(pos_end)
                    # Add the prob mass to this cell
                    cell_end.prob_spread += prob_jump/num_pt
                    # Add this cell to the cell list, if it is not there
                    if not (cell_end in cell_list_distributed):
                        cell_list_distributed.append(cell_end)
                
            
            
            # Find the second index of Q
            # This should be the index corresponding to the current cell.
            # Meaning of Qij is the prob of j-->i.
            index_Q_2 = self.map_labelpair_indexP[cell.label]
            # Spread the probs to Q
            self.distribute_Q(index_Q_2, cell_list_distributed)
                
            # Reset prob_spread for each cell in cell_list_distributed
            for c in cell_list_distributed:
                c.prob_spread = 0
           
      
               
    def pos_to_cell_dim2(self, pos):
        # Find the cell that contains the given position 'pos'
        x, y = pos
        dx, dy = self.dx
        label_x = int(np.floor(x/dx))
        label_y = int(np.floor(y/dy))
        label = (label_x, label_y)
        index = self.map_labelpair_indexP[label]
        return self.cells[index]
    
    
    
    def cal_Q_1(self, num_tau_invervals):
        # Calculate Q in 1D case
        dt = self.dt 
        dtau = dt/num_tau_invervals # dtau further divides dt. It is used to deal with uniform distribution of jumps within dt.
        # Get the base pulse rate
        pulse_rate_base = self.pulses[0].rate # Currently, there is only a single pulse object.
        # Get the base pulse amp
        pulse_amp_base = self.pulses[0].amp
        pulse_amp = self.pulses[0].amp
        for cell in self.cells:
            # Get the starting position (the center of the current cell)
            pos_start = cell.pos
            # Update the state-dependent pulse rate and amp
            # This is the crucial part. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pulse_rate = pulse_rate_base/(1+0*cell.pos[0])
            
            # Deterministic part
            # Calculate the end position
            pos_end = self.cal_pos_deterministic_n(ICs=pos_start, duration=dt)
            # Calculate the total prob to be spread
            prob_deterministic = 1 - pulse_rate * dt
            # Find the cells to be spread
            cell_list_distributed = self.distribute(pos_end, prob_deterministic)
            # Find the second index of Q
            # This should be the index corresponding to the current cell.
            # Meaning of Qij is the prob of reaching i from j.?????
            index_Q_2 = self.map_labelpair_indexP[cell.label]
            # Spread the probs to Q
            self.distribute_Q(index_Q_2, cell_list_distributed)
            
            #if cell.label == (91,):
            #    print("cell_list_distributed, deterministic")
            #    print("Starting position", cell.pos)
            #    print("end position", pos_end)
            #    print("prob spread total", prob_deterministic)
            #    for cell_temp in cell_list_distributed:
            #        print("cell_list_distributed", cell_temp.label, "prob spread", cell_temp.prob_spread)
            #    print(" ")   
            
            
            # Jump part
            # Assume that the pulse arrives in the middle of each dtau
            for tau in np.arange(dtau/2, dt, dtau):
                # The jump in the first neurite length
                amp_jump = [pulse_amp,]
                # Calculate the final postion after dt
                pos_end = self.cal_pos_jump_n(ICs=pos_start, duration=dt, t_jump=tau, amp_jump=amp_jump)
                # The prob mass to be spread
                prob_jump = pulse_rate * dtau # In fact, this is pulse_rate*dt *dtau/dt
                # Get the cell list where the prob mass to be spread to
                cell_list_distributed = self.distribute(pos_end, prob_jump)
                
                self.distribute_Q(index_Q_2, cell_list_distributed)
            
                #if cell.label == (91,):
                #    print("cell_list_distributed, jump")
                #    print("Starting position", cell.pos)
                #    print("end position", pos_end)
                #    print("prob spread", prob_jump)
                #    for cell_temp in cell_list_distributed:
                #        print("cell_list_distributed", cell_temp.label, "prob spread", cell_temp.prob_spread)
                #    print(" ") 
            
            
    def distribute_Q(self, index_Q_2, cell_list_distributed): 
        # Assign the probs of cell list to entries of Q
        # Note that index_Q_2 is the starting state
        # index_Q_1 is the end state
        for cell_distributed in cell_list_distributed:
            index_Q_1 = self.map_labelpair_indexP[cell_distributed.label]
            self.Q[index_Q_1][index_Q_2] += cell_distributed.prob_spread
         
            
    def cal_pos_deterministic_n(self, ICs, duration):
        # Calculate the position after 'duration', starting from ICs
        # There is no jump within 'duration'.
        sol = odeint(self.RHS_deterministic_n, y0 = ICs, t=[0, duration])
        pos_end = []
        for d in range(self.dim):
            pos_end.append(sol[-1][d])
        
        # If the new position is outside the region, just return the initial conditions
        if self.list_compare_less_equal(pos_end, self.xmax):
            return pos_end
        else:
            return ICs
    
    
    def cal_pos_jump_n(self, ICs, duration, t_jump, amp_jump):
        # Calculate the position after 'duration', starting from ICs, 
        # when there is a jump at t_jump
        # Calculate the position before jump
        sol = odeint(self.RHS_deterministic_n, y0 = ICs, t=[0, t_jump])
        
        # Apply the jump
        ICs_new = []
        for d in range(self.dim):
            ICs_new.append(sol[-1][d]+amp_jump[d])
        
        # Use the position after jump as initial conditions
        sol = odeint(self.RHS_deterministic_n, y0 = ICs_new, t=[t_jump, duration])
        
        # Store the final position
        pos_end = []
        for d in range(self.dim):
            pos_end.append(sol[-1][d])
        
        # If the final position is outside the region, just return the initial conditions.
        if self.list_compare_less_equal(pos_end, self.xmax):
            return pos_end
        else:
            return ICs
    
    
    def list_compare_less_equal(self, l1, l2):
        # Compare two lists to see whether elements in the first list are less or equal to the corresponding elements in the second list.
        # This is used in the functions to calculate the deterministic motion and the motion with a jump
        for a1, a2 in zip(l1, l2):
            if a1>a2:
                return False
        return True
    
    
    
    
    def RHS_deterministic_n(self, y, t):
        # This function gives the right-hand side of each equation
        # This is a crucial function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        
        RHS = []
        
        for d in range(self.dim):
            # Get the base parameter values
            beta = self.neurites[d].beta
            Kg = self.neurites[d].Kg
            ng = self.neurites[d].ng
            r = self.neurites[d].r
            
            # For a single neurite
            if self.dim==1:
                deri = beta * Hill(y[d], Kg, ng) - r * y[d]
            
            # For 2 neurites
            #deri = beta * Hill(y[d], Kg, ng) - r*(1+1*y[1-d]) * y[d] # For mutual inhibition through the retraction rate
            #deri = beta * Hill(y[d], Kg, ng) - r*(1+0.3*y[1-d])/(1+0.3*Hill(y[d],K=1,n=2)) * y[d]
            #deri = beta * Hill(y[d], Kg, ng) - r*(1+1*Heavi(np.max([y[d], y[1-d]]), 3)*(np.max([y[d], y[1-d]])-3)) * y[d] # inhibition from itself is included
            
            # exponential decay of inhibitory substance
            #deri = beta * Hill(y[d], Kg, ng) - r*(1+0.5*(y[d]+y[1-d])*np.exp(-0.9*y[d]))* y[d]
            # Simple decay of inhibitory substance
            if self.dim==2:
                # Crucial codes to change the rates with lengths
                # Unquote to use
                # No change, base rates
                #deri = beta/(1+0*(y[d]+y[1-d])/(1+0*y[d])) * Hill(y[d], Kg, ng) - r*(1+0.0*(y[d]+y[1-d])/(1+0.0*y[d])) * y[d] # inhibition from itself is included
                # Test the deterministic part
                #deri = 0 
                # Change the retraction rates
                #deri = beta/(1+0*(y[d]+y[1-d])/(1+0*y[d])) * Hill(y[d], Kg, ng) - r*(1+0.02*(y[d]+y[1-d])/(1+0.0*y[d])) * y[d] # inhibition from itself is included
                # For test of bistability
                #deri = beta/(1+0*(y[d]+y[1-d])/(1+0*y[d])) * Hill(y[d], Kg, n=1) - r*(1+2.52*(y[d]+y[1-d])/(1+3.9*y[d])) * y[d]
                # Combination of the retraction rate and pulse rate
                #deri = beta/(1+0*(y[d]+y[1-d])/(1+0*y[d])) * Hill(y[d], Kg, ng) - r*(1+0.5*(y[d]+y[1-d])/(1+3.5*y[d])) * y[d] # inhibition from itself is included
                
                alpha = 0.01
                deri = beta * Hill(y[d], Kg, ng) - r*(1+alpha*(y[d]+y[1-d])) * y[d] # inhibition from itself is included

                
            RHS.append(deri)
        
        return RHS
    
    
    def distance_square_dim2(self, x, y):
        # calculate the distance between x and y, which are both vectors
        # This is Eucledian distance
        return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    
    def distance_square_dim1(self, x, y):
        # calculate the distance between x and y
        return np.abs(x-y)
    
    
    def distribute(self, pos, prob_total=1):
        # Distribute the probability mass ``prob_total'' to the target position ``pos''
        
        # Create a list for the cells to be spread prob mass
        list_cell_spread = []
        
        # 2D case
        if self.dim == 2:
            # Extract the target positions
            x, y = pos
            # Cell sizes in the two directions
            dx, dy = self.dx
            # Maximum indices in the two direction
            label_max_x, label_max_y = self.index_max
                
            
            # Find the label of the nearest lattice point
            # Note that this is not the same as a cell label
            # The bottom-left corner has lattice label (0,0), 
            # while the top-right corner has (self.n_cells[0], self.n_cells[1])
                        
            nx = int(np.ceil(x/dx - 0.5))
            ny = int(np.ceil(y/dy - 0.5))
            
            # Sum of inverse distances to the nearest cells
            d_inv_total = 0
            # Spread (unnormalized probabilities)
            for i in range(nx-1, nx+1): #i=nx-1, nx
                for j in range(ny-1, ny+1): #j=ny-1, ny
                    if i>=0 and j>=0 and i<=label_max_x and j<=label_max_y: # check whether (i,j) is a valid cell label pair
                        # find the index of the cell    
                        index_cell = self.map_labelpair_indexP[(i,j)]
                        # Get the cell
                        cell = self.cells[index_cell]
                        # Calculate the distance of the cell center to the target position
                        # This might be zero, so we add 1e-8 to it to prevent 1/0 when calculating 1/d
                        d = self.distance_square_dim2(cell.pos, pos) + 1e-8
                        d_inv_total += 1/d
                        # Spread the probability (unnormalized)
                        # The closer the distance, the higher the proportion of probability
                        cell.prob_spread = prob_total*1/d
                        # Put the cell into the list
                        list_cell_spread.append(cell)
            
            # Rescale the prob to be spread
            for cell in list_cell_spread:
                cell.prob_spread = cell.prob_spread/d_inv_total
        
        # 1D case
        if self.dim == 1:
            # Extract the target positions
            x, = pos
            # Cell sizes in the two directions
            dx, = self.dx
            # Maximum indices in the two direction
            label_max_x, = self.index_max
                
            
           
            nx = int(np.ceil(x/dx - 0.5))
           
            
            # Sum of inverse distances to the nearest cells
            d_inv_total = 0
            # Spread (unnormalized probabilities)
            for i in range(nx-1, nx+1): #i=nx-1, nx
                if i>=0 and i<=label_max_x: # check whether (i,j) is a valid cell label pair
                    # find the index of the cell    
                    index_cell = self.map_labelpair_indexP[(i,)]
                    # Get the cell
                    cell = self.cells[index_cell]
                    # Calculate the distance of the cell center to the target position
                    # This might be zero, so we add 1e-8 to it to prevent 1/0 when calculating 1/d
                    d = self.distance_square_dim1(cell.pos[0], pos[0]) + 1e-8
                    d_inv_total += 1/d
                    # Spread the probability (unnormalized)
                    # The closer the distance, the higher the proportion of probability
                    cell.prob_spread = prob_total*1/d
                    # Put the cell into the list
                    list_cell_spread.append(cell)
            
            # Rescale the prob to be spread
            for cell in list_cell_spread:
                cell.prob_spread = cell.prob_spread/d_inv_total
        
            
        # These codes spread the probability according to a discrete form of the delta function.
        # This method introduces a large fake diffusion and is currently abandoned.
        '''
        # Distribute the probability mass ``prob_total'' to the target position ``pos''
        # Create lists to store mim and max indices for cells to be distributed to.
        index_min = []
        index_max = []
        
        # To simplify codes
        dx = self.dx
        xmin = self.xmin
        
        for d in range(self.dim):
            # Calculate the smallest possible index
            ind_temp = int(np.ceil((pos[d]-xmin[d])/dx[d] -2 -0.5))
            # Compare it with the lowest acceptale index
            ind_temp = max(ind_temp, 0)
            # Add the final result to the list
            index_min.append( ind_temp )
            
            # Calculate the largest possible index
            ind_temp = int(np.floor((pos[d]-xmin[d])/dx[d] +2 -0.5))
            # Compare it with the largest acceptale index
            ind_temp = min(ind_temp, self.index_max[d])
            # Add the final result to the list
            index_max.append( ind_temp )
        
       
        
        # Create a list for the cells to be spread prob mass
        list_cell_spread = []
        # A sum of probs for rescaling
        prob_sum = 0
        
        # spread prob mass in 2D case
        if self.dim == 2:
            for n1 in range(index_min[0], index_max[0] + 1):
                for n2 in range(index_min[1], index_max[1] + 1):
                    # Get the index of a cell in self.cells
                    n = self.map_labelpair_indexP[(n1,n2)]
                    # Get the cell
                    cell = self.cells[n]
                    # Spread the probability
                    cell.prob_spread = delta(cell.pos[0]-pos[0], dx[0]) * delta(cell.pos[1]-pos[1], dx[1])
                    prob_sum += cell.prob_spread
                    list_cell_spread.append(cell)
        
        # spread prob mass in 1D case
        if self.dim == 1:
            for n1 in range(index_min[0], index_max[0] + 1):
                # Get the index of a cell in self.cells
                n = self.map_labelpair_indexP[(n1,)]
                # Get the cell
                cell = self.cells[n]
                # Spread the probability
                cell.prob_spread = delta(cell.pos[0]-pos[0], dx[0])
                prob_sum += cell.prob_spread
                list_cell_spread.append(cell)
        
       
                
        for cell in list_cell_spread:
            # Rescale the prob to be spread
            # For a target position near boundaries, only part of the support of the discrete delta is used.
            # Therefore, prob_sum may not be 1
            # Furthermore, the above codes do not include prob_total. They actually only spread a prob mass of 1.
            cell.prob_spread = cell.prob_spread/prob_sum * prob_total
           
        # Return the list of cells
        '''
        
        return list_cell_spread
    
  
    
  
    
    
    
    
    
    
   
    def check_Q_sum_to_1(self, Q):
        # check whether each column in the input Q is summed to 1
        # The input Q can be the original self.Q, or iterated self.Q_iter and so on.
        # This is used to checked whehter Q is correctly generated.
        self.column_sums_Q = []
        index_list = [] # Used for plotting
        for n in range(self.N):
            index_list.append(n)
            self.column_sums_Q.append(np.sum(Q[:,n]))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(index_list, self.column_sums_Q)
        
       
        ax.set_ylabel(r'Q column sum')
        ax.set_xlabel('index')
        
        # This is used to prevent the yscale from becoming exponential, such as 1e-12+1, which is essentially 1
        ax.ticklabel_format(useOffset=False)

        
     
    def store_Q(self, filename):
        # Write the Q matrix to a file for further Markov chain analysis in Matlab
        
        N = self.N
        
        # Open the file and write data
        with open(filename, 'w') as f_obj:
            # The first line stores the dimension of Q, i.e, N
            #f_obj.write(str(self.N)+'\n')
            
            # Write the entries of Q
            # Note that by the convention of GCM, the 2nd index of Q is the starting state
            # But in matlab, the 1st index is the starting state
            for j in range(N):
                for i in range(N):
                    f_obj.write(str(self.Q[i,j])+' ')
                f_obj.write('\n')
                
    def read_Q(self, filename):
        # Read the stored Q
        # Note that the stored Q follows the convention in math textbooks
        # Qij is the prob of i-->j
        # But here in GCM, Qij is the prob of j-->i
        
        # Open the file and read data
        with open(filename) as file_obj:
            lines = file_obj.readlines()
            
        # Auxillary indices
        i = 0 # Current row in the file (ith line in lines_read)
        j = 0 # Current column in the line (jth element in the current line)
        
        # If the stored matrix does not match the current settng, quit.
        line_test = lines[0]
        line_test_str= (line_test.rstrip()).split()
        if len(line_test_str) != self.N:
            print("Not match, failed")
            return "Failed"
        
        for line in lines:
            # First get rid of '\n' at the end of each line and split the remaining string into parts.
            line_str= (line.rstrip()).split()
            
            # Reset j
            j = 0
            for entry in line_str:
                self.Q[j,i] = float(entry)
                j += 1
                
            i += 1    
  
    
    
    
    def cal_P(self, duration):
        # Calculate P from the current time
        n_iter = int(duration/self.dt)
        for n in range(n_iter):
            self.P1 = np.dot(self.Q, self.P1)
            self.t += self.dt
        print("Current time:", self.t)
        
    def cal_P_from_t0(self, duration):
        # Calculate P from t=0
        self.P1 = np.copy(self.P0)
        n_iter = int(duration/self.dt)
        self.t = 0
        for n in range(n_iter):
            self.P1 = np.dot(self.Q, self.P1)
            self.t += self.dt
        print("Current time:", self.t)
        
    def cal_P_from_t0_exp(self, n_iter):
        # Calculate P by progressing exponentially
        # Calculate Q, Q^2, Q^4,...
        self.P1 = np.copy(self.P0)
        self.t = 2**n_iter * self.dt # The first iteration gives (Q^2)*P
        self.Q_2expn = np.copy(self.Q)
        for n in range(n_iter):
            self.Q_2expn = np.dot(self.Q_2expn, self.Q_2expn)
        self.P1 = np.dot(self.Q_2expn, self.P0)
        print("Current time:", self.t)
        
          
        
    def cal_Q_eigen(self):
        # Find eigenvalues and eigenvectors of Q
        self.Q_evals, self.Q_evects = LA.eig(self.Q)
        
        # Calculate the inverse of the eigenvector matrix
        # There is some problem here: Q_evects is singular
        # Therefore the following line is abandoned.
        #self.Q_evects_inv = LA.inv(self.Q_evects)
            
    def cal_Ps(self):
        # Calculate the stationary distribution
        # The stationary distribution corresponds to the eigenvector of the largest eigenvalue.
    
        # Use w and v to simplify notations
        w = self.Q_evals
        v = self.Q_evects

        len_w = len(w)
        eval_max = 0
        ind_eval_max = 0
        
        for index in range(len_w):
            eval_real = np.real(w[index])
            #print("eval_real:", eval_real)
            if eval_real >= eval_max:
                eval_max = eval_real
                ind_eval_max = index
                
        print("The maximum eigenvalue: ", w[ind_eval_max])
        
        const_normalizing = np.sum(v[:,ind_eval_max])
        self.Ps = np.real(v[:,ind_eval_max]/const_normalizing)
        
    
    def convert_P_to_Pmat(self, P, P_mat):
        # Convert the probability list into a matrix for visualization
        for n in range(self.N):
            i, j = self.map_indexP_labelpair[n]
            P_mat[i][j] = P[n]
            
    def store_P(self, P, filename):
        # Store distribution for plotting in matlab
        # Format of each line: label_x, label_y, pos_x, pos_y, P
        # Open the file and write data
        with open(filename, 'w') as f_obj:
            for n in range(self.N):
                label_x, label_y = self.map_indexP_labelpair[n]
                pos_x, pos_y = self.cells[n].pos
                string_written = str(label_x) + ' ' + str(label_y) + ' '
                string_written += str(pos_x) + ' ' + str(pos_y) + ' '
                string_written += str(P[n]) + '\n'
                f_obj.write(string_written)
    
            
    def show_Q(self, Q):
        # Plot Q
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax = fig.add_subplot(111, projection='3d')
        n_row, n_col = Q.shape
        x = np.arange(0, n_col)
        y = np.arange(0, n_row)
        X, Y = np.meshgrid(x,y)
        im = ax.pcolormesh(X, Y, Q)
        if Q is self.Q:
            ax.set_title('Q')
        elif Q is self.Q_n:
            ax.set_title('Q_n')
        fig.colorbar(im, ax=ax)
        #ax.plot_surface(X, Y, self.Q)
        
    def test_Q_symmetry(self):
        # Test the ``symmetry'' of Q
        # Used to test codes
        # Not needed anymore
        Q = self.Q
        N = self.N
        # A sum storing the accumulated asymmetry in Q
        asymmetry_accumulate = 0
        for i in range(N):
            for j in range(N):
                Qij = Q[i,j]
                # Reverse the label pairs corresponding to i and j
                labelpair1 = self.map_indexP_labelpair[i]
                labelpair2 = self.map_indexP_labelpair[j]
                
                labelpair1_reversed = (labelpair1[1], labelpair1[0])
                labelpair2_reversed = (labelpair2[1], labelpair2[0])
                           
                s = self.map_labelpair_indexP[labelpair1_reversed]
                r = self.map_labelpair_indexP[labelpair2_reversed]
                Qsr = Q[s,r]
                
                asymmetry_accumulate += np.abs(Qij-Qsr)
            
        print("asymmetry_accumulate=", asymmetry_accumulate)
        
    def set_pos_mesh(self):
        # Set up the position mesh for the colormaps of P_mat and Ps_mat
        xmin = self.xmin
        #xmax = self.xmax
        dx = self.dx
        x = np.arange(xmin[0]+0.5*dx[0], xmin[0] + (0.5+self.index_max[0]+1)*dx[0], dx[0])
        y = np.arange(xmin[1]+0.5*dx[1], xmin[1] + (0.5+self.index_max[1]+1)*dx[1], dx[1])
        self.X, self.Y = np.meshgrid(x,y, indexing='ij')
        
        
        
    def show_P_mat(self, P_mat):
        # Plot the colormap of a given probability matrix P_mat
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        self.set_pos_mesh()
        
        P_mat_max = np.max(P_mat)
        im = ax.pcolormesh(self.X, self.Y, P_mat, vmin=0, vmax=P_mat_max)
        
        # Depending on the input P_mat, set different titles.
        if P_mat is self.P0_mat:
            title = r'$P_0(l_1, l_2)$'
            title += ' ' + 't={}'.format(round(self.t, 2))   
        elif P_mat is self.P1_mat:
            title = r'$P(l_1, l_2)$'
            title += ' ' + 't={}'.format(round(self.t, 2))   
        elif P_mat is self.Ps_mat:
            title = r'$P_s(l_1, l_2)$'
        else:
            title = ''
        
        
        
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        
        ax.set_xlabel(r'$l_1$')
        ax.set_ylabel(r'$l_2$')
        
        ax.set_xlim([0,10])
        ax.set_ylim([0,10])
        
        
    def save_P_mat(self, P_mat, filename):
        # Save the given P_mat together with X and Y mesh in an excel file
        
        # Create pandas dataframes
        df_P = pd.DataFrame(P_mat)
        df_X = pd.DataFrame(self.X)
        df_Y = pd.DataFrame(self.Y)
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        
        # Write each DataFrame to a specific sheet
        df_P.to_excel(writer, sheet_name="probability", index=False)
        df_X.to_excel(writer, sheet_name="X_mesh", index=False)
        df_Y.to_excel(writer, sheet_name="Y_mesh", index=False)
        
        #close the Pandas Excel writer and output the Excel file
        writer.save()
     

    def show_P_dim1(self, P):
        # Show the probability P in 1D case
        
        pos_list = []
        pos = 0.0
        for n in range(len(P)):
            pos += self.dx[0]
            pos_list.append(pos)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if P is self.Ps:
            ax.plot(pos_list, P)
            ax.set_ylabel(r'$P_s(l)$')
        elif P is self.P1:
            ax.plot(pos_list, P, label='t={}'.format(round(self.t, 2)))
            ax.legend()
            ax.set_ylabel(r'$P(l)$')
        elif P is self.P0:
            ax.plot(pos_list, P)
            ax.legend()
            ax.set_ylabel(r'$P_0(l)$')
        
        
        ax.set_xlabel(r'$l$')
        
        #beta = self.neurites[0].beta
        #Kg = self.neurites[0].Kg
        #ng = self.neurites[0].ng
        #r = self.neurites[0].r
        
        #xmin = np.min(pos_list)
        #xmax = np.max(pos_list)
        #ymin = 0
        #ymax = 0.01
        
        #ax.set_ylim([ymin,ymax])
        #ax.text(0.02*(xmax-xmin)+xmin, 0.9*(ymax-ymin)+ymin, r"$\beta={},  K_g={:.2f},  n_g={},  r={}$".format(beta, Kg, ng, r))

    
        
    def FPT_1(self, x_ust, t_iter):
        # Find the index of the first cell in the absorbing region in the x1 direction
        index_1_ust = int(np.ceil(((x_ust-self.xmin[0])/self.dx[0])-0.5))
        
        F_T_prev =  self.cal_P_absorbing_1(index_1_ust)
        
        T_mean = 0
       
        for n in range(1, t_iter+1):
            if n==1:
                self.P1 = np.dot(self.Q, self.P0)
            else:
                self.P1 = np.dot(self.Q, self.P1)
            F_T = self.cal_P_absorbing_1(index_1_ust)
            dF_T = F_T - F_T_prev
            #print('density:', dF_T/self.dt)
            T_mean += n*self.dt * dF_T
            #print('F_T', F_T)
            #print('F_T_prev', F_T_prev)
            #if np.abs(dF_T)<=1e-4:
            #    break
            F_T_prev = F_T
        print('T_mean=', T_mean)
        return T_mean
        
    def cal_P_absorbing_1(self, index_1_ust):
        P_absorbing = 0
        for cell in self.cells:
            if cell.label[0]>=index_1_ust:
                P_absorbing += self.P1[self.map_labelpair_indexP[cell.label]]
        #print("P_absorbing=", P_absorbing)
        return P_absorbing
    
    
    def show_P_dim2(self, P):
        # Plot the P list directly without converting it to a matrix.
        length = len(P)
        index_list = np.arange(0, length)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(index_list, P)
        if P is self.Ps:
            y_label = r'$P_s$'
        elif P is self.P1:
            y_label = r'$P$'
        elif P is self.P0:
            y_label = r'$P_0$'
        else:
            y_label = ''
        ax.set_ylabel(y_label)
        ax.set_xlabel('index')
        
        
    
        
    def find_cell_given_position_2D(self, pos):
        # Given a position for the two-neurite case, find the nearest cell
        # ``pos'' in the function argument should be input as [x_position, y_position]
        # The nearest cell is just the one that contains the target position
        x, y = pos
        dx, dy = self.dx
        index_x = int(np.floor(x/dx))
        index_y = int(np.floor(y/dy))
        
        if index_x < 0 or index_y < 0 or index_x > self.index_max[0] or index_y>self.index_max[1]:
            print("Invalid position!")
        
        return [index_x, index_y]
        
    '''
    def recurrence_pos(self, pos):
        # Given a position, determine the recurrency of the cell that contains the position
        # This function does not work well, abandoned.
        index_x, index_y = self.find_cell_given_position_2D(pos) # Find the label pair of the cell.
        index_cell = self.map_labelpair_indexP[(index_x, index_y)] # Find the index of the cell in Q
        
        Q_sum = 0 # Q_sum = Q_ii^(1) + Q_ii^(2) + .....
        n_iter = 200000 # #terms in Q_sum
        
        evals = self.Q_evals
        evects = self.Q_evects
        
        for n in range(1, n_iter+1):
            if n % 1000 ==0:
                print("n=", n)
            for s in range(self.N): # s runs through all cells
                Q_sum += (evals[s]**n)*evects[index_cell,s]*evects[index_cell,s]
        
        print("Q_sum=", Q_sum)
    '''   
    '''    
    def convert_sparse(self):
        # Convert matrices and vectors into sparse forms
        # It seems that using a sparse matrix does not improve efficiency
        # Abandoned.
        self.Q_sparse = csr_matrix(self.Q)
    '''   
    
    def cal_Q_n(self, n_iter=1000, continue_or_not=0):
        # Calculate Q^n by using sparse matrix multiplication
        # continue_or_not==0: calculate from n=0
        # continue_or_not==1: continue calculation the last result
        # Abandoned because of poor performance
        if continue_or_not == 0:
            Q_n = self.Q_sparse
            for i in range(n_iter-1):
                print(i)
                Q_n = Q_n.dot(self.Q_sparse)
        
            self.Q_n = Q_n.toarray()
        
        if continue_or_not == 1:
            Q_n = csr_matrix(self.Q_n)
            for i in range(n_iter):
                print(i)
                Q_n = Q_n.dot(self.Q_sparse)
        
        
        
    '''   
    def cal_Q_inf(self):
        # Calculate various things related to Q^{inf}
        N = self.N
        self.Q_inf = np.zeros((N,N))
        # Create a list for diagonal entries
        # Used to plot
        Q_inf_ii_list = []
        
        for i in range(N):
            for j in range(N):
                self.Q_inf[i,j] = self.Ps[i]*self.Ps[j]
            Q_inf_ii_list.append(self.Q_inf[i,i])
        
        # Show Q_inf_ii
        index_list = np.arange(0, N)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(index_list, Q_inf_ii_list)
        ax.set_ylabel(r'$Q_{ii}^{\infty}$')
        ax.set_xlabel('index')
        
        # Plot Q_inf in 2D case
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n_row, n_col = self.Q_inf.shape
        x = np.arange(0, n_col)
        y = np.arange(0, n_row)
        X, Y = np.meshgrid(x,y)
        im = ax.pcolormesh(X, Y, self.Q_inf)
        ax.set_title(r'$Q_{ii}^{\infty}$')
        fig.colorbar(im, ax=ax)
    '''
    
    
    
    
    def FPT_find_mode(self):
        # Find the mode of the stationary distribution in the lower right region
        value_max = 0
        for i in range(self.index_max[0]+1):
            for j in range(i+1):
                if self.Ps_mat[i,j]>value_max:
                    value_max = self.Ps_mat[i,j]
                    label_max = (i,j)
        # Find the index correpsonding to this label
        index_max = self.map_labelpair_indexP[label_max]
        # Return the center position of this cell
        return self.cells[index_max].pos
        
    
    
    
    def inrange(self, x, xmin, xmax):
        # Decide whether x is between xmin and xmax
        # Used in FPT_set_absorbing_BC_Q
        if x>=xmin and x<=xmax:
            return True
        else:
            return False
    
    def FPT_set_absorbing_BC(self, pars):
        # Old codes for setting absorbing BCs
        # Modify Q to implement absorbing boudary conditions
        # Extract the center and widths of the absorbing region
        # The format of pars: [[center_0, center_1],[width_0, width_1]]
        center, width = pars # center and width are vectors
        
        # Cells inside the absorbing region
        cells_absorbing = []
        
        # Initialized Q_absorb
        self.Q_absorb = np.copy(self.Q)
        
        # Modify Q
        for cell in self.cells:
            # If the cell is within the absorbing region
            if self.inrange(cell.pos[0], center[0]-width[0]/2, center[0]+width[0]/2) and self.inrange(cell.pos[1], center[1]-width[1]/2, center[1]+width[1]/2):
                # Put the cell in the list
                cells_absorbing.append(cell)
                # Find the index of the cell according to its label
                index_cell = self.map_labelpair_indexP[cell.label]
                # Q[index_cell, index_cell] is set to 1, while all other cells with the same 2nd index are set to 0.
                for n in range(self.N):
                    self.Q_absorb[n, index_cell] = 0
                self.Q_absorb[index_cell, index_cell] = 1
        
        # Return the cell list, used in FPT_cal
        return cells_absorbing
    
    
    def sum_prob_cells(self, cell_list):
        # Sum the current probabilities of the given cells
        P_sum = 0
        for cell in cell_list:
            index_cell = self.map_labelpair_indexP[cell.label]
            P_sum += self.P1[index_cell]
            
        return P_sum           
        
    
    
    def FPT_cal(self, n_iter, width):
        # Old codes for FPT
        # Calculate the first passage time from an initial postion to the target absorbing region
        # The width is input manually. It depends on observation of the distribution
        # Find the location of the mode/peak in the absorbing region
        pos_lower_mode = self.FPT_find_mode()
        print("pos_lower_mode=", pos_lower_mode)
        # Initial position is the reverse of the mode position above
        pos_init = [pos_lower_mode[1], pos_lower_mode[0]]
        print("pos_init=", pos_init)
        
        # Set up initial distribution
        self.set_P(pos_init)
        
        # Modify Q and return the cell list in the absorbing region
        cells_absorbing = self.FPT_set_absorbing_BC([pos_lower_mode, width])
        
        
        # Distribution of the first passage time at the last step
        FT_last = 0
        # Distribution of the first passage time at the current step
        FT = 0
        # Mean first passage time
        T_mean = 0
        
      
        # Main loop
        for n in range(1, n_iter+1):
            # Calculate the current prob distribution
            self.P1 = np.dot(self.Q_absorb, self.P1)
            # Find the total probability in the absorbing region
            FT = self.sum_prob_cells(cells_absorbing)
            # Add the current contribution to the mean FPT
            T_mean += n * self.dt * (FT - FT_last)
            # Update FT_last
            FT_last = FT
        
        print("MFPT=", T_mean)
        
     
        
    def check_inside_target_region(self, pos, target_region):
        # Check whether a position 'pos' is inside the target region
        # 'target_region' may contain multiple subregions
        # target_region = [subregion1_list, subregion2_list,...]
        
        # This label marks whether pos is inside target_region
        inside_or_not = False
        # Check each subregion
        # As long as pos is in one subregion, return True
        for subregion in target_region:
            # Extract boundaries of the subregion
            # Each subregion is assumed to be rectangle
            xmin, xmax, ymin, ymax = subregion
            if self.inrange(pos[0], xmin, xmax) and self.inrange(pos[1], ymin, ymax):
                inside_or_not = True
                return inside_or_not
        
        return inside_or_not
            
        
        
    def FPT_cal_Q_absorb(self, target_region):
        # Modify Q to implement absorbing boudary conditions
        # 'target_region' may contain multiple subregions
        # target_region = [subregion1_list, subregion2_list,...]
        
        # Cells inside the absorbing region
        cells_absorbing = []
        
        # The modified Q is stored in self.Q_absorb        
        self.Q_absorb = np.copy(self.Q)
        
        # Modify Q
        for cell in self.cells:
            # If the cell is within the absorbing region
            if self.check_inside_target_region(cell.pos, target_region):
                # Put the cell in the list
                cells_absorbing.append(cell)
                # Find the index of the cell according to its label
                index_cell = self.map_labelpair_indexP[cell.label]
                # Q[index_cell, index_cell] is set to 1, while all other cells with the same 2nd index are set to 0.
                for n in range(self.N):
                    self.Q_absorb[n, index_cell] = 0
                self.Q_absorb[index_cell, index_cell] = 1
        
        # Return the cell list, used in FPT_cal
        return cells_absorbing
        
    
    def FPT_cal_exp(self, pos_init, target_region, n_iter):
        # Estimate the first passage time from an initial postion to the target absorbing region
        # The powers of Q are used.
        # pos_init = [pos_x, pos_y]
        # target_region = [xmin, xmax, ymin, ymax].  It depends on observation of the distribution
                
        # Set up initial distribution
        self.set_P_not_spread(pos_init)
        
        # Modify Q and return the cell list in the absorbing region
        cells_absorbing = self.FPT_cal_Q_absorb(target_region)
        
        # Distribution of the first passage time at the previous step
        mass_absorbed_last = 0
        # Distribution of the first passage time at the current step
        mass_absorbed = 0
        # Upper and lower estimations of FPT
        T_upper = 0
        T_lower = 0
        
        # lists for plotting
        mass_absorbed_list = []
        t_list = []
        n_list = []
        
        # Used to store the power of Q_absorb
        Q_absorb_iter = self.Q_absorb
        
        # Main loop
        for n in range(1, n_iter+1):
            print("n=",n)
            # Calculate the current prob distribution
            Q_absorb_iter = np.dot(Q_absorb_iter, Q_absorb_iter)
            # Calculate the current prob distribution
            # Note that P0 is used here.
            self.P1 = np.dot(Q_absorb_iter, self.P0)
            # Find the total probability in the absorbing region
            mass_absorbed = self.sum_prob_cells(cells_absorbing)
            # Add the current contribution
            # Calculate the upper and lower sum
            T_upper += (2**n) * self.dt * (mass_absorbed - mass_absorbed_last)
            T_lower += (2**(n-1)) * self.dt * (mass_absorbed - mass_absorbed_last)
            # Update mass_absorbed_last
            mass_absorbed_last = mass_absorbed
            # Store the probability, time and #iterations
            mass_absorbed_list.append(mass_absorbed)
            t_list.append((2**n) * self.dt)
            n_list.append(n)
        
        # Take average
        T_mean = 0.5 * (T_upper + T_lower)
        print("T_mean=", T_mean)
        
        
        # Plot the evolution of the absorbed probability mass
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # The x axis is the #iterations
        ax.plot(n_list, mass_absorbed_list)
        # Return the three estimations of FPT for further use.      
        return [T_mean, T_lower, T_upper]
    
    
    def FPT_cal_increment(self, pos_init, target_region, power_incr, n_iter):
        # Calculate FPT by using constant increment in steps
        
        
        
        # Set up initial distribution
        self.set_P_not_spread(pos_init)
        
        # Modify Q and return the cell list in the absorbing region
        cells_absorbing = self.FPT_cal_Q_absorb(target_region)
        
        # Calculate Q_absorb_iter^power_incr
        Q_absorb_iter = np.copy(self.Q_absorb)
        for i in range(power_incr-1):
            Q_absorb_iter = np.dot(Q_absorb_iter, self.Q_absorb)
        
        # Distribution of the first passage time at the previous step
        mass_absorbed_last = 0
        # Distribution of the first passage time at the current step
        mass_absorbed = 0
        # Upper and lower estimations of FPT
        T_upper = 0
        T_lower = 0
        
        # lists for plotting
        mass_absorbed_list = []
        t_list = []
        n_list = []
        
               
        # Main loop
        for n in range(1, n_iter+1):
            print("n=",n)
            # Calculate the current prob distribution
            self.P1 = np.dot(Q_absorb_iter, self.P1)
            # Find the total probability in the absorbing region
            mass_absorbed = self.sum_prob_cells(cells_absorbing)
            # Add the current contribution
            # Calculate the upper and lower sum
            T_upper += (n*power_incr) * self.dt * (mass_absorbed - mass_absorbed_last)
            T_lower += ((n-1)*power_incr) * self.dt * (mass_absorbed - mass_absorbed_last)
            # Update mass_absorbed_last
            mass_absorbed_last = mass_absorbed
            # Store the probability, time and #iterations
            mass_absorbed_list.append(mass_absorbed)
            t_list.append(n * power_incr * self.dt)
            n_list.append(n)
        
        # Take average
        T_mean = 0.5 * (T_upper + T_lower)
        print("T_mean=", T_mean)
        
        
        # Plot the evolution of the absorbed probability mass
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # The x axis is the #iterations
        ax.plot(n_list, mass_absorbed_list)
        # Return the three estimations of FPT for further use.      
        return [T_mean, T_lower, T_upper]
            
    
    def cal_prob_mass_target_region(self, P, target_region):
        # Calculate the probability mass in a target region
        
        
        
        # First get the cells in the target region
        cell_list = []
        for cell in self.cells:
            if self.check_inside_target_region(cell.pos, target_region):
                cell_list.append(cell)
        
        # Sum the probabilities
        prob_mass = 0
        for cell in cell_list:
            indexP = self.map_labelpair_indexP[cell.label]
            prob_mass += P[indexP]
            
        print("prob_mass=", prob_mass)
        return prob_mass
    
    
###################################################################
####################################################################
####################################################################   


    def eps_committor(self, eps, target_region, pos_init):
        # Stability in finite time using epsilon committor introduced by Lindner
        
        # Get the transpose of Q
        self.Q_T = np.transpose(self.Q)
        M = self.Q_T # This is for consistency with Lindner's paper
        
        # Set up I_A
        # The given target region is rectangular
        # target_region = [xmin, xmax, ymin, ymax]
        I_A = np.zeros((self.N, 1))
        
        xmin, xmax, ymin, ymax = target_region
        for cell in self.cells:
            x, y = cell.pos
            if x>=xmin and x<=xmax and y>=ymin and y<=ymax:
                index = self.map_labelpair_indexP[cell.label]
                I_A[index] = 1
        
                
        #self.show_P_dim2(I_A)
        #I_A_mat = np.zeros((self.index_max[0]+1, self.index_max[1]+1))
        #self.convert_P_to_Pmat(I_A, I_A_mat)
        #self.show_P_mat(I_A_mat)
                
        
        
        # Find the index of the cell containing the initial position
        index_init = self.map_labelpair_indexP[self.pos_to_cell_dim2(pos_init).label]
        print("index_init", index_init)
        
        
        # Create a unit vector for calculating the eps-committor
        I = np.identity(self.N)
        #self.show_P_dim2(I)
        
        
        
        # Calculate the eps-committor
        
        print("det(I-(1-eps)*M)", "{:e}".format(LA.det(I-(1-eps)*M)))
        self.q_eps = eps * np.dot(LA.inv(I-(1-eps)*M), I_A)
        
        print("q_eps shape", self.q_eps.shape)
        
        
      
        # Return the stability measured by the component of q_eps corresponding the the given initial position
        return self.q_eps[index_init][0] # [0] is used here because q_eps[index_init] is one row in q_eps and it is an array of a single entry        
        
    
    def eps_committor_list(self, power_list, target_region, pos_init):
        # Generate eps_committor list for plotting
        b_eps_list = []
        
        for n in power_list:
            eps = np.power(10,-n)
            b_eps = self.eps_committor(eps, target_region, pos_init)
            b_eps_list.append(b_eps)
        
        return b_eps_list

     

    def eps_committor_all(self, eps, target_region):
        # Stability in finite time using epsilon committor introduced by Lindner
        
        # Get the transpose of Q
        self.Q_T = np.transpose(self.Q)
        M = self.Q_T # This is for consistency with Lindner's paper
        
        # Set up I_A
        # The given target region is rectangular
        # target_region = [xmin, xmax, ymin, ymax]
        I_A = np.zeros((self.N, 1))
        
        xmin, xmax, ymin, ymax = target_region
        for cell in self.cells:
            x, y = cell.pos
            if x>=xmin and x<=xmax and y>=ymin and y<=ymax:
                index = self.map_labelpair_indexP[cell.label]
                I_A[index] = 1
        
        
        # Create a unit vector for calculating the eps-committor
        I = np.identity(self.N)
        
        # Calculate the eps-committor
        #print("det(I-(1-eps)*M)", "{:e}".format(LA.det(I-(1-eps)*M)))
        self.q_eps = eps * np.dot(LA.inv(I-(1-eps)*M), I_A)
        
        # We flatten q_eps here to turn it into a 1darray
        # q_eps is a column vector, i.e., 2darray, which it not compatible with self.convert_P_to_Pmat 
        # We hope to reuse self.convert_P_to_Pmat, so we flatten q_eps
        q_eps_reshape = self.q_eps.flatten()  
        # Initialize q_eps_field
        self.q_eps_field = np.zeros((self.index_max[0]+1, self.index_max[1]+1))
        # Convert
        self.convert_P_to_Pmat(q_eps_reshape, self.q_eps_field)
        
        # Calculate the mean dwell time
        # mean dwell time = eps-committor/eps
        self.mean_dwell_time = self.q_eps_field/eps
        
        
        # Return the stability measured by the component of q_eps corresponding the the given initial position
        #return self.q_eps        
    
    def show_eps_committor_field(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        self.set_pos_mesh()
        
              
        im = ax.pcolormesh(self.X, self.Y, self.q_eps_field, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        
        ax.set_title(r'$\epsilon$-committor')
        
        ax.set_xlabel(r'$l_1$')
        ax.set_ylabel(r'$l_2$')
        
        ax.set_xlim([0,10])
        ax.set_ylim([0,10])
        
        # Show the mean dwell time in the target region        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        im = ax.pcolormesh(self.X, self.Y, self.mean_dwell_time)
        fig.colorbar(im, ax=ax)
        
        ax.set_title(r'Mean dwell time')
        
        ax.set_xlabel(r'$l_1$')
        ax.set_ylabel(r'$l_2$')
        
        ax.set_xlim([0,10])
        ax.set_ylim([0,10])   
        
        

####################################################################
####################################################################
####################################################################
dt=0.1
d_spatial=0.2

probs = Probs(dim=2)
probs.set_dt(dt)
probs.set_range(xmin=[0,0], xmax=[10,10], dx=[d_spatial, d_spatial])
probs.create_cells_n()
probs.init_P_Q()
probs.set_P_not_spread(pos_init=[0,0])
probs.create_neurites_n(pars_all_neurites=[[10, np.sqrt(21), 2, 1], [10, np.sqrt(21), 2, 1]])
amp=1.0
rate=1.0
probs.create_pulse(pars=[amp, rate])

probs.neurites[0].plot_growth_retraction_terms()

# Calculate Q
# If using previous result, directly run the reading function, see below
#probs.cal_Q(num_tau_invervals=3)
#probs.cal_Q(num_tau_invervals=1)

# Calculate Q using random sampling
#probs.cal_Q_random(num_pt=100, num_tau_invervals=3)
#probs.cal_Q_random(num_pt=100, num_tau_invervals=1)


# Store Q
#probs.store_Q("../GCM_data/data_Q.txt")

# Read Q
probs.read_Q("../GCM_data/data_Q.txt") # Remember to change dx and dt when reading Q

# Check row sums of Q
probs.check_Q_sum_to_1(Q=probs.Q)

#probs.show_Q(probs.Q)

# Show the initial distribution
#probs.convert_P_to_Pmat(probs.P0, probs.P0_mat)
#probs.show_P_mat(probs.P0_mat)

'''
# Find evals evects of Q
probs.cal_Q_eigen()


# Stationary distribution
probs.cal_Ps()
probs.convert_P_to_Pmat(probs.Ps, probs.Ps_mat)
probs.show_P_mat(probs.Ps_mat)
probs.show_P_dim2(P=probs.Ps)
'''

'''
# Calculation from t=0
probs.cal_P_from_t0(duration=10)
probs.convert_P_to_Pmat(probs.P1, probs.P1_mat)
probs.show_P_mat(probs.P1_mat)

# Calculation from the last step
probs.cal_P(duration=1000)
probs.convert_P_to_Pmat(probs.P1, probs.P1_mat)
probs.show_P_mat(probs.P1_mat)

# Calculation by exponentially iterating Q
probs.cal_P_from_t0_exp(n_iter=15)
probs.convert_P_to_Pmat(probs.P1, probs.P1_mat)
probs.show_P_mat(probs.P1_mat)


# Save a given P_mat
probs.save_P_mat(probs.P1_mat, "../GCM_data/distribution/pulseamp/bimodal/bimodal.xlsx")
probs.save_P_mat(probs.P1_mat, "../GCM_data/distribution/pulseamp/two-to-one/two-to-one.xlsx")
probs.save_P_mat(probs.P1_mat, "../GCM_data/distribution/pulserate/two-to-one/two-to-one.xlsx")
probs.save_P_mat(probs.P1_mat, "../GCM_data/distribution/retraction/bimodal/bimodal.xlsx")

'''


'''
# FPT
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,6], target_region=[5,8,0,1], n_iter=50)
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,0], target_region=[7,11,7,11], n_iter=50)

T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,0], target_region=[0,1.5,7,11], n_iter=50)
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0.1,7.3], target_region=[7,11,7,11], n_iter=50)




T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,0], target_region=[[5,10,0,2],[0,2,5,10]], n_iter=30)
T_mean, T_lower, T_upper = probs.FPT_cal_increment(pos_init=[0,0], target_region=[[5,10,0,2],[0,2,5,10]], power_incr=100, n_iter=4000)

T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,0], target_region=[[5,10,0,3],[0,3,5,10]], n_iter=30)

T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,0], target_region=[[0,1,6,9],[6,9,0,1]], n_iter=30)



# For mean transition time
# For flipping
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,6], target_region=[[4,8,0,2],], n_iter=15)
# For polarization
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,0], target_region=[[0,2,6,10],[6,10,0,2]], n_iter=30)
# For 2axon
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,0], target_region=[[6,10,6,10],], n_iter=30)
# For 1axon to 2axon
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,8], target_region=[[6,10,6,10],], n_iter=30)

# For polarization (retraction, rate=2.5)
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[0,0], target_region=[[4,8,0,2],[0,2,4,8]], n_iter=30)


# Mean escape time for flipping, escape from the equilibrium in the middle
T_mean, T_lower, T_upper = probs.FPT_cal_exp(pos_init=[5,5], target_region=[[0,10,0,4],[0,10,8,10],[0,4,4,8],[8,10,4,8]], n_iter=10)

'''





'''
# Calculate the probability mass in a given region
# Remember to first calculate P1
# For pulse rate and amp
prob_mass_0axon = probs.cal_prob_mass_target_region(P=probs.P1, target_region=[[0,2,0,2]])
prob_mass_1axon = probs.cal_prob_mass_target_region(P=probs.P1, target_region=[[0,2,6,10],[6,10,0,2]])
prob_mass_2axon = probs.cal_prob_mass_target_region(P=probs.P1, target_region=[[6,10,6,10]])



# For retraction, rate=2.5
prob_mass_0axon = probs.cal_prob_mass_target_region(P=probs.P1, target_region=[[0,2,0,2]])
prob_mass_1axon = probs.cal_prob_mass_target_region(P=probs.P1, target_region=[[0,2,6,10],[6,10,0,2]])
prob_mass_2axon = probs.cal_prob_mass_target_region(P=probs.P1, target_region=[[3,10,3,10]])





'''


'''
# Stability
# target_region and pos_init may be different for different mechanisms
target_region=[0,3,3,10]
# pos_init=[0,7] # For pulse rate and amp
pos_init = [0,6] # For retraction

power_list = np.arange(0.1, 11, 0.1, dtype=float)
b_eps_list = probs.eps_committor_list(power_list, target_region, pos_init)

eps = 1e-1
b_eps = probs.eps_committor(eps, target_region, pos_init)
print('b_eps=',b_eps)
'''

'''
# eps-committor field
target_region=[0,3,3,10]
eps = 1e-8
probs.eps_committor_all(eps, target_region)
probs.show_eps_committor_field()
'''




'''
# Single neurite
h = 0.02
probs = Probs(dim=1)
probs.set_dt(h)
probs.set_range(xmin=[0,], xmax=[10,], dx=[h,])
probs.create_cells_n()
probs.init_P_Q()
probs.set_P(pos_init=[0,])
probs.create_neurites_n(pars_all_neurites=[[10, np.sqrt(21), 2, 1], ])
#probs.create_neurites_n(pars_all_neurites=[[6, 2*np.sqrt(2), 2, 1], ])
probs.neurites[0].plot_growth_retraction_terms()
probs.create_pulse(pars=[1.0, 1.0])
#print(probs.cal_pos_deterministic_n(ICs=[1.9,], duration=10))
#print(probs.cal_pos_jump_n(ICs=[1.5,], duration=0.8, t_jump=0.4, amp_jump=[1,]))
#list_cell_spread = probs.distribute(pos=[2,])
probs.cal_Q_1(num_tau_invervals=3)
# Find evals evects of Q
probs.cal_Q_eigen()
# Check row sums of Q
probs.check_Q_sum_to_1()

#probs.FPT_1(x_ust=2.0, t_iter=int(300/h))
probs.FPT_1(x_ust=3.0, t_iter=int(200/h))
'''

'''
probs.cal_P_from_t0(duration=1)
probs.show_P_dim1(P=probs.P0)
probs.cal_P(duration=10)
probs.show_P_dim1(P=probs.P1)
'''

'''
probs.cal_Ps()
probs.show_P_dim1(P=probs.Ps)
#probs.show_P_dim1(P=probs.P0)
#probs.cal_P_from_t0(duration=10)
#probs.show_P_dim1(P=probs.P1)
'''

