import numpy as np
from scipy.signal import cont2discrete

class BCIAdmittanceController:
    """
    Event-Triggered Virtual Neuromuscular Admittance Controller
    for MI-BCI Maze Navigation.
    """
    def __init__(self):
        # ---------------------------------------------------------
        # 1. DESIGN CONSTANTS (Recommended Initial Parameter Set)
        # ---------------------------------------------------------
        self.dt = 0.2          # EEG update period in seconds
        self.tau_c = 0.90      # Classifier confidence threshold 
        self.alpha_a = 0.283   # Virtual muscle activation gain (T_a = 0.6s) 
        self.g_m = 5.0         # Gain mapping activation imbalance to admittance input 
        self.z_on = 0.5        # Turn-commitment threshold 
        
        # Turn maneuver kinematics (User-defined for the specific robot)
        self.omega_T = 0.5     # Fixed turning rate during T_R / T_L in rad/s

        # Continuous Admittance Parameters 
        M_I = 1.0              # Virtual inertia 
        D_I = 4.0              # Virtual damping (zeta = 1.0) 
        K_I = 4.0              # Virtual stiffness (omega_n = 2.0 rad/s) 

        # ---------------------------------------------------------
        # 2. STATE SPACE SETUP & DISCRETIZATION
        # ---------------------------------------------------------
        # Continuous-time model: M_I * z_ddot + D_I * z_dot + K_I * z = f_I
        # State vector xi = [z_I, z_dot_I]^T
        A_c = np.array([
            [0, 1],
            [-K_I / M_I, -D_I / M_I]
        ])
        B_c = np.array([
            [0],
            [1 / M_I]
        ])
        C_c = np.eye(2)
        D_c = np.zeros((2, 1))

        # Zero-order-hold (ZOH) discretization to get A_I and B_I 
        sys_d = cont2discrete((A_c, B_c, C_c, D_c), self.dt, method='zoh')
        self.A_I = sys_d[0]
        self.B_I = sys_d[1]

        # ---------------------------------------------------------
        # 3. INTERNAL STATE INITIALIZATION
        # ---------------------------------------------------------
        self.a_R = 0.0                 # Right virtual muscle activation
        self.a_L = 0.0                 # Left virtual muscle activation
        self.xi_I = np.zeros((2, 1))   # Intent state [z_I(k), z_dot_I(k)]^T
        self.q_state = 'F'             # Finite State Machine (FSM) state: F, D, T_L, T_R, H

    def step(self, p_L, p_R, d_f, d_th, heading_aligned=False):
        """
        Executes one discrete time step (k) of the BCI navigation algorithm.
        
        Inputs:
        - p_L, p_R: Classifier probabilities for Left/Right MI (must sum to 1).
        - d_f: Forward distance to nearest obstacle.
        - d_th: Decision-distance threshold.
        - heading_aligned: Boolean indicating if a turn maneuver is complete.
        
        Returns:
        - omega: The commanded angular velocity.
        - current_state: The current FSM state (for logging/debugging).
        - z_I: The current accumulated intent (for plotting).
        """
        
        # --- ALGORITHM STEP 4: Detect Decision Region & FSM Updates ---
        # Detect decision region chi_D 
        chi_D = 1 if d_f <= d_th else 0
        
        # FSM State Transitions
        if self.q_state == 'F' and chi_D == 1:
            self.q_state = 'D' # Enter Decision State
            
        if heading_aligned and self.q_state in ['T_L', 'T_R']:
            self.q_state = 'F' # Turn completed, return to Forward

        # --- ALGORITHM STEPS 1 & 2: Confidence-Vetted Evidence ---
        e_v = 0.0
        if max(p_L, p_R) >= self.tau_c:
            e_v = p_R - p_L 
            
        # --- ALGORITHM STEP 3: Virtual Muscle Activations ---
        # Rectify the evidence into left/right channels 
        u_R = max(e_v, 0.0)
        u_L = max(-e_v, 0.0)
        
        # Update first-order activation dynamics 
        self.a_R = (1 - self.alpha_a) * self.a_R + self.alpha_a * u_R
        self.a_L = (1 - self.alpha_a) * self.a_L + self.alpha_a * u_L
        
        # Net virtual steering effort 
        f_I = self.g_m * (self.a_R - self.a_L)

        # --- ALGORITHM STEP 5: Event-Triggered Admittance Dynamics ---
        # Event gate sigma(k) is 1 ONLY in state D and chi_D == 1 
        sigma = 1.0 if (self.q_state == 'D' and chi_D == 1) else 0.0
        
        if sigma == 1.0:
            # Active admittance response to BCI 
            self.xi_I = self.A_I @ self.xi_I + self.B_I * f_I
        else:
            # Passive decay outside decision regions (zero input) 
            self.xi_I = self.A_I @ self.xi_I
            
        z_I = self.xi_I[0, 0] # Extract position state (accumulated intent)

        # --- ALGORITHM STEPS 6, 7 & 9: Discrete Turn Commitment & Robot Command ---
        omega_command = 0.0
        
        if self.q_state == 'D':
            # Check commitment boundaries 
            if z_I >= self.z_on:
                self.q_state = 'T_R'
            elif z_I <= -self.z_on:
                self.q_state = 'T_L'
            # Note: Feasible-action filtering (Section II-F) would go here 
            # to override T_R/T_L to HOLD if a wall is blocking that direction.

        # Output turn velocities based on FSM state 
        if self.q_state == 'T_R':
            omega_command = -self.omega_T # Right turn
        elif self.q_state == 'T_L':
            omega_command = self.omega_T  # Left turn
        elif self.q_state == 'F':
            omega_command = 0.0 # Autonomous forward velocity (v_A) handles this
            
        return omega_command, self.q_state, z_I