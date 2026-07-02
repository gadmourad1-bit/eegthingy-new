"""
claude generated code:
=============================================================================
BCI-HRI Virtual Neuromuscular Admittance: Omega (omega) Generator
=============================================================================
Based on: "Event-Triggered Virtual Neuromuscular Admittance for Discrete
          MI-BCI Maze Navigation"
 
PIPELINE:
    [pL(k), pR(k)]
        |
        v
    Confidence gate              ->  ev(k)                  [Eq. 8]
        |
        v
    Virtual agonist-antagonist
    muscle activation            ->  uR, uL, aR(k), aL(k)  [Eq. 10-12]
        |
        v
    Net virtual steering effort  ->  fI(k)                  [Eq. 14]
        |
        v
    Event-triggered admittance
    dynamics                     ->  xi_I(k) = [zI, zIdot]  [Eq. 18-19]
        |
        v
    Turn commitment rule         ->  omega(k)               [Eq. 20-21]
 
HOW TO USE:
    from bci_omega_generator import compute_omega
    import numpy as np
 
    aR, aL = 0.0, 0.0       # initialise ONCE before your loop
    xi     = np.zeros(2)
 
    # at every EEG window timestep k:
    omega, command, aR, aL, xi, ev = compute_omega(
        pL=0.07, pR=0.93,   # from your classifier
        aR=aR, aL=aL,       # carry over from previous step
        xi=xi,              # carry over from previous step
        sigma=1             # 1 if robot is at a maze junction, else 0
    )
    # omega   -> send to robot as angular velocity [rad/s]
    # command -> 'RIGHT', 'LEFT', or 'HOLD'
    # aR, aL, xi -> pass back in on the NEXT timestep
=============================================================================
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm   # matrix exponential for exact ZOH
 
 
# =============================================================================
# SECTION 1 - DESIGN CONSTANTS  (Section III recommended parameter set)
# =============================================================================
 
# EEG update period [s] — one new probability vector every DELTA_T seconds
DELTA_T   = 0.2          # Eq. (25): 5 Hz classifier output rate
 
# Confidence gate threshold tau_c
# Windows where max(pL,pR) < TAU_C are treated as idle and produce ev=0
TAU_C     = 0.90         # Eq. (26)
 
# Virtual muscle activation time-constant Ta [s]
# Must satisfy:  DELTA_T < Ta < T_MAX_D   [Eq. 27]
# Larger Ta = smoother but slower;  smaller Ta = more reactive but noisier
T_A       = 0.6          # Eq. (28)
 
# Discrete activation gain alpha_a = 1 - exp(-Delta_t / Ta)   [Eq. 13]
# This is the EXACT discrete-time equivalent of a first-order RC filter
ALPHA_A   = 1.0 - np.exp(-DELTA_T / T_A)   # approx 0.283  per Eq. (29)
 
# Admittance model: virtual inertia, damping ratio, settling time
M_I     = 1.0   # virtual inertia MI    — Eq. (31), set to 1 without loss of generality
ZETA    = 1.0   # damping ratio  zeta   — Eq. (32), 1.0 = critically damped (no overshoot)
T_S     = 2.0   # desired settling time — Eq. (35), good starting point for maze nav
 
# Derive natural frequency, stiffness, and damping from the above [Eq. 33-34]
OMEGA_N = 4.0 * ZETA / T_S              # natural frequency omega_n = 2 rad/s
K_I     = M_I * OMEGA_N**2              # virtual stiffness KI = 4
D_I     = 2.0 * ZETA * M_I * OMEGA_N   # virtual damping   DI = 4
 
# Virtual muscle-to-force gain gm
# Chosen so steady-state admittance displacement zss = gm * e_acc_bar / KI = 1
# for a typical accepted evidence value — this normalises the output   [Eq. 42]
E_ACC_BAR = 0.8          # conservative lower bound on accepted |ev|    [Eq. 38]
G_M       = K_I / E_ACC_BAR   # = 5
 
# Turn-commitment boundary z_on   [Eq. 43]
# Robot commits to a turn only when |zI| >= Z_ON
Z_ON    = 0.5
 
# Maximum allowed dwell time in decision state [s]   [Eq. 52]
T_MAX_D = 3.0
 
# Physical angular velocity used during an actual turn manoeuvre [rad/s]
# This is omega_T in Eq. (21) — tune to your robot's kinematics
OMEGA_T = 0.5
 
 
# =============================================================================
# SECTION 2 - EXACT ZOH DISCRETISATION OF THE ADMITTANCE MODEL
# =============================================================================
#
# Continuous-time admittance model (Eq. 17):
#   MI * z_Iddot + DI * z_Idot + KI * zI = sigma(t) * fI(t)
#
# State-space form with  xi_I = [zI, z_Idot]^T:
#   xi_Idot = Ac * xi_I + Bc * sigma * fI
#
#   Ac = [[ 0,      1    ],     Bc = [[ 0   ],
#          [-KI/MI, -DI/MI]]          [ 1/MI ]]
#
# Exact ZOH discretisation via the matrix-exponential augmented-matrix trick:
#
#   M_aug = [[ Ac,  Bc ],    (3x3 matrix)
#             [ 0,   0  ]]
#
#   expm(M_aug * Delta_t) = [[ AI,  BI ],
#                             [  0,   1 ]]
#
#   AI (2x2) = discrete state matrix
#   BI (2x1) = discrete input matrix
#
# This is EXACT — no approximation error from Euler integration.
 
def _build_zoh_matrices(M, D, K, dt):
    """
    Compute the ZOH discrete-time matrices AI and BI.
 
    Parameters
    ----------
    M  : float  -- virtual inertia   MI
    D  : float  -- virtual damping   DI
    K  : float  -- virtual stiffness KI
    dt : float  -- sampling period   Delta_t [s]
 
    Returns
    -------
    A_d : (2,2) ndarray  -- discrete state matrix  AI
    B_d : (2,1) ndarray  -- discrete input matrix  BI
    """
    Ac  = np.array([[0.0,   1.0 ],
                    [-K/M, -D/M]])
    Bc  = np.array([[0.0  ],
                    [1.0/M]])
    n, m = 2, 1
    aug  = np.zeros((n + m, n + m))
    aug[:n, :n] = Ac          # top-left:  Ac
    aug[:n, n:] = Bc          # top-right: Bc
    # bottom row stays zero
    eM = expm(aug * dt)       # matrix exponential
    return eM[:n, :n], eM[:n, n:]   # return AI, BI
 
 
# Pre-compute once at module load — constant throughout the run
AI, BI = _build_zoh_matrices(M_I, D_I, K_I, DELTA_T)
 
 
# =============================================================================
# SECTION 3 - PIPELINE FUNCTIONS  (one function per equation block)
# =============================================================================
 
def compute_signed_evidence(pL, pR, tau_c=TAU_C):
    """
    STEP 1 — Confidence-vetted signed EEG evidence  ev(k)   [Eq. 8]
 
    Only trust the classifier when one class clearly dominates.
    Ambiguous or idle windows (neither class >= tau_c) produce ev=0,
    so they do NOT excite the intent dynamics — critical for reliability.
 
    Any accepted nonzero evidence satisfies |ev| >= 2*tau_c-1 = 0.8  [Eq. 9]
 
    Parameters
    ----------
    pL    : float [0,1]  -- probability of LEFT  motor-imagery
    pR    : float [0,1]  -- probability of RIGHT motor-imagery
    tau_c : float        -- confidence gate threshold
 
    Returns
    -------
    ev : float
> 0  -> RIGHT intent
< 0  -> LEFT  intent
         = 0  -> idle / ambiguous (gate closed)
    """
    assert abs(pL + pR - 1.0) < 1e-6, "pL + pR must equal 1.0"
 
    if max(pL, pR) >= tau_c:   # gate OPEN: high-confidence window
        ev = pR - pL
    else:
        ev = 0.0               # gate CLOSED: discard ambiguous window
    return ev
 
 
def update_muscle_activations(aR, aL, ev, alpha_a=ALPHA_A):
    """
    STEP 2 — Virtual agonist-antagonist muscle activation  [Eq. 10-12]
 
    Why this matters:
      Raw ev(k) can flip sign from one noisy EEG window to the next.
      Directly using it as a command would cause chattering.
      Instead, we mimic biological muscle:
        1. RECTIFY ev into two non-negative directional channels (Eq. 10)
        2. SMOOTH each with a first-order IIR filter (Eq. 11-12)
      This requires SUSTAINED evidence before fI builds up to threshold.
 
    Parameters
    ----------
    aR, aL  : float  -- current activations  aR(k), aL(k) in [0,1]
    ev      : float  -- signed evidence from Step 1
    alpha_a : float  -- activation gain = 1 - exp(-Dt/Ta)
                        small alpha_a = slow smooth response
                        large alpha_a = fast reactive response
 
    Returns
    -------
    aR_new, aL_new : float  -- updated activations aR(k+1), aL(k+1)
    """
    uR = max( ev, 0.0)   # [ev]+  : right excitation  [Eq. 10]
    uL = max(-ev, 0.0)   # [-ev]+ : left  excitation  [Eq. 10]
 
    aR_new = (1.0 - alpha_a) * aR + alpha_a * uR   # Eq. (11)
    aL_new = (1.0 - alpha_a) * aL + alpha_a * uL   # Eq. (12)
    return aR_new, aL_new
 
 
def compute_steering_effort(aR, aL, gm=G_M):
    """
    STEP 3 — Net virtual steering effort  fI(k)   [Eq. 14]
 
    The activation imbalance (aR - aL) is scaled by gm to produce
    the scalar force input to the admittance model.
    gm is chosen so that a sustained high-confidence command gives
    a steady-state displacement of 1 (normalised output).
 
    Parameters
    ----------
    aR, aL : float  -- RIGHT and LEFT muscle activations
    gm     : float  -- gain mapping activation imbalance to force
 
    Returns
    -------
    fI : float  -- net steering effort (+ve = right, -ve = left)
    """
    return gm * (aR - aL)   # Eq. (14)
 
 
def update_admittance_state(xi, fI, sigma, A=AI, B=BI):
    """
    STEP 4 — Admittance state propagation  [Eq. 18-19]
 
    xi_I(k+1) = AI * xi_I(k) + BI * sigma(k) * fI(k)   [Eq. 18]
 
    THE EVENT GATE sigma(k) is the central innovation of the paper:
      sigma = 1  ->  robot is at a maze junction (decision region)
                     admittance is ACTIVE; evidence accumulates in zI
      sigma = 0  ->  robot is in a corridor (autonomous forward motion)
                     input is zero; zI decays naturally to zero
                     NO heuristic reset (zI <- 0) is ever needed because
                     AI is Schur-stable (eigenvalues inside unit circle)
 
    Parameters
    ----------
    xi    : (2,) ndarray  -- current state [zI(k), zIdot(k)]
    fI    : float         -- net steering effort from Step 3
    sigma : int {0,1}     -- event gate
    A, B  : ndarrays      -- pre-computed ZOH matrices
 
    Returns
    -------
    xi_new : (2,) ndarray  -- updated state [zI(k+1), zIdot(k+1)]
    """
    xi_col = xi.reshape(2, 1)
    xi_new = A @ xi_col + B * (sigma * fI)   # Eq. (18)
    return xi_new.flatten()
 
 
def decide_turn_command(zI, z_on=Z_ON):
    """
    STEP 5 — Discrete turn commitment rule  [Eq. 20]
 
    Compare admittance displacement zI against the boundary +/- z_on.
    A turn fires only after persistent high-confidence evidence has
    built up enough to push zI across the boundary.
    Single noisy EEG windows cannot trigger a turn — the admittance
    dynamics act as a temporal integrator/filter before commitment.
 
    Parameters
    ----------
    zI    : float  -- admittance displacement (position state)
    z_on  : float  -- commitment boundary
 
    Returns
    -------
    command : str    -- 'RIGHT', 'LEFT', or 'HOLD'
    omega   : float  -- angular velocity command omega(k) [rad/s]
    """
    if   zI >=  z_on:  return 'RIGHT', +OMEGA_T   # Eq. (21)
    elif zI <= -z_on:  return 'LEFT',  -OMEGA_T   # Eq. (21)
    else:              return 'HOLD',   0.0        # still accumulating
 
 
def compute_omega(pL, pR, aR, aL, xi, sigma,
                  tau_c=TAU_C, alpha_a=ALPHA_A, gm=G_M, z_on=Z_ON):
    """
    MASTER FUNCTION — full single-timestep pipeline
    ================================================
    Input:  classifier probabilities + carry-over states
    Output: omega(k) angular velocity command + updated states
 
    Call this in your real-time loop. Always pass the returned
    (aR_new, aL_new, xi_new) back in as inputs on the next call.
 
    Parameters
    ----------
    pL, pR   : float        -- left/right probabilities (sum to 1)
    aR, aL   : float        -- muscle activations from PREVIOUS timestep
    xi       : (2,) ndarray -- admittance state   from PREVIOUS timestep
    sigma    : int {0,1}    -- 1 if robot is at a junction, else 0
 
    Returns
    -------
    omega   : float         -- angular velocity command [rad/s]
    command : str           -- 'RIGHT', 'LEFT', or 'HOLD'
    aR_new  : float         -- RIGHT activation  (pass to next timestep)
    aL_new  : float         -- LEFT  activation  (pass to next timestep)
    xi_new  : (2,) ndarray  -- admittance state  (pass to next timestep)
    ev      : float         -- signed evidence   (for logging)
    """
    ev             = compute_signed_evidence(pL, pR, tau_c)           # Step 1
    aR_new, aL_new = update_muscle_activations(aR, aL, ev, alpha_a)  # Step 2
    fI             = compute_steering_effort(aR_new, aL_new, gm)     # Step 3
    xi_new         = update_admittance_state(xi, fI, sigma)          # Step 4
    command, omega = decide_turn_command(xi_new[0], z_on)            # Step 5
    return omega, command, aR_new, aL_new, xi_new, ev
 
 
# =============================================================================
# SECTION 4 - SIMULATION DEMO
# =============================================================================
#
# Scenario:
#   k =  0- 9 : corridor, sigma=0, ambiguous EEG (pL=pR=0.5) -> ignored
#   k = 10-29 : junction, sigma=1, user imagines RIGHT (pR=0.93)
#               zI grows until it crosses +Z_ON -> RIGHT turn committed
#   k = 30-49 : turn done, sigma=0, state decays naturally to zero
#
 
def simulate(prob_sequence, sigma_sequence):
    """Run the pipeline over a full sequence. Returns dict of signal logs."""
    aR, aL = 0.0, 0.0
    xi     = np.zeros(2)
    logs   = {k: [] for k in ('ev','aR','aL','fI','zI','omega','command')}
 
    for (pL, pR), sigma in zip(prob_sequence, sigma_sequence):
        omega, cmd, aR, aL, xi, ev = compute_omega(pL, pR, aR, aL, xi, sigma)
        logs['ev'].append(ev);  logs['aR'].append(aR);  logs['aL'].append(aL)
        logs['fI'].append(compute_steering_effort(aR, aL))
        logs['zI'].append(xi[0])
        logs['omega'].append(omega);  logs['command'].append(cmd)
    return logs
 
 
if __name__ == "__main__":
    N         = 50
    prob_seq  = [(0.5,0.5)]*10 + [(0.07,0.93)]*20 + [(0.5,0.5)]*20
    sigma_seq = [0]*10          + [1]*20            + [0]*20
 
    results = simulate(prob_seq, sigma_seq)
    time    = np.arange(N) * DELTA_T
 
    first_right = next((k for k,c in enumerate(results['command']) if c=='RIGHT'), None)
    print(f"RIGHT turn committed at step k={first_right}, t={first_right*DELTA_T:.1f}s")
 
    fig, axes = plt.subplots(5, 1, figsize=(11,13), sharex=True)
    fig.suptitle("BCI-HRI Pipeline: Classifier Probabilities -> omega(k)", fontsize=13, fontweight='bold')
 
    axes[0].plot(time, results['ev'], color='steelblue')
    axes[0].axhline(0, color='gray', lw=0.7, ls='--')
    axes[0].set_ylabel('ev(k)'); axes[0].set_title('Step 1 - Signed evidence [Eq.8]')
    axes[0].grid(True, alpha=0.3)
 
    axes[1].plot(time, results['aR'], color='tomato',     label='aR RIGHT')
    axes[1].plot(time, results['aL'], color='dodgerblue', label='aL LEFT')
    axes[1].set_ylabel('Activation'); axes[1].set_title('Step 2 - Muscle activations [Eq.11-12]')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
 
    axes[2].plot(time, results['fI'], color='darkorange')
    axes[2].axhline(0, color='gray', lw=0.7, ls='--')
    axes[2].set_ylabel('fI(k)'); axes[2].set_title('Step 3 - Steering effort [Eq.14]')
    axes[2].grid(True, alpha=0.3)
 
    axes[3].plot(time, results['zI'], color='purple', lw=2)
    axes[3].axhline( Z_ON, color='red',  lw=1.5, ls='--', label=f'+z_on={Z_ON}')
    axes[3].axhline(-Z_ON, color='blue', lw=1.5, ls='--', label=f'-z_on=-{Z_ON}')
    axes[3].axhline(0, color='gray', lw=0.7, ls='--')
    axes[3].set_ylabel('zI(k)'); axes[3].set_title('Step 4 - Admittance intent state [Eq.18-19]')
    axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)
 
    axes[4].step(time, results['omega'], where='post', color='darkgreen', lw=2)
    axes[4].set_ylabel('omega(k) [rad/s]'); axes[4].set_xlabel('Time [s]')
    axes[4].set_title('Step 5 - Angular velocity omega(k) [Eq.20-21]')
    axes[4].grid(True, alpha=0.3)
 
    for ax in axes:
        ax.axvspan(10*DELTA_T, 29*DELTA_T, alpha=0.08, color='orange')
 
    plt.tight_layout()
    plt.savefig('omega_pipeline.png', dpi=150, bbox_inches='tight')
    plt.show()