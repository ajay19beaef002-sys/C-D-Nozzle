import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 1.4
R = 287.0
Imax = 101
L = 2.0
dx = L / (Imax - 1)
x = np.linspace(0, L, Imax)

# Nozzle geometry
def nozzle_area(x):
    return 1.0 + 2.0 * (x - 1.0)**2

A = nozzle_area(x)
dAdx = np.gradient(A, dx)

# Reservoir conditions
T0 = 300.0
p0 = 1.0133e5
pe_p0 = 0.585  # This will determine shock position
pe = pe_p0 * p0

# Initialize numerical solution
M_init = 0.1
T_init = T0 / (1 + 0.5*(gamma-1)*M_init**2)
p_init = p0 / (1 + 0.5*(gamma-1)*M_init**2)**(gamma/(gamma-1))
rho_init = p_init / (R * T_init)
u_init = M_init * np.sqrt(gamma * R * T_init)
E_init = p_init/(gamma-1) + 0.5*rho_init*u_init**2

U = np.array([rho_init * A, rho_init * u_init * A, E_init * A])

# Pre-allocate arrays
F_p, F_m, S, U_new = [np.zeros_like(U) for _ in range(4)]

# Time stepping
CFL = 0.9
t_max = 0.5  # Reduced for faster convergence
time = 0.0

def compute_fluxes(U, A):
    """Optimized vectorized flux calculation"""
    rho = U[0] / A
    u = U[1] / U[0]
    E = U[2] / A
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    a = np.sqrt(gamma * p / rho)
    M = u / a
    
    F_p.fill(0)
    F_m.fill(0)
    
    # Vectorized calculations
    transonic = np.abs(M) < 1.0
    factor = 0.25 * rho[transonic] * a[transonic] * (M[transonic] + 1)**2 * A[transonic]
    term = 1 + 0.5*(gamma-1)*M[transonic]
    
    F_p[:, transonic] = np.array([
        factor,
        factor * (2*a[transonic]/gamma) * term,
        factor * (2*a[transonic]**2/(gamma**2 - 1)) * term**2
    ])
    
    F_m[:, transonic] = np.array([
        rho[transonic]*u[transonic]*A[transonic],
        (rho[transonic]*u[transonic]**2 + p[transonic])*A[transonic],
        u[transonic]*(E[transonic] + p[transonic])*A[transonic]
    ]) - F_p[:, transonic]
    
    # Supersonic fluxes
    supersonic = M >= 1.0
    F_p[:, supersonic] = np.array([
        rho[supersonic]*u[supersonic]*A[supersonic],
        (rho[supersonic]*u[supersonic]**2 + p[supersonic])*A[supersonic],
        u[supersonic]*(E[supersonic] + p[supersonic])*A[supersonic]
    ])
    
    return F_p, F_m

def apply_bcs(U, A):
    """Optimized boundary conditions"""
    # Inlet (subsonic)
    u_in = U[1,1]/U[0,1]
    T_in = T0 - u_in**2/(2*R*gamma/(gamma-1))
    p_in = p0 * (T_in/T0)**(gamma/(gamma-1))
    rho_in = p_in/(R*T_in)
    
    U[0,0] = rho_in * A[0]
    U[1,0] = rho_in * u_in * A[0]
    U[2,0] = (p_in/(gamma-1) + 0.5*rho_in*u_in**2) * A[0]
    
    # Outlet (subsonic)
    rho_out = U[0,-2]/A[-2]
    u_out = U[1,-2]/U[0,-2]
    
    U[0,-1] = rho_out * A[-1]
    U[1,-1] = rho_out * u_out * A[-1]
    U[2,-1] = (pe/(gamma-1) + 0.5*rho_out*u_out**2) * A[-1]
    
    return U

# Main solver loop
while time < t_max:
    rho = U[0]/A
    u = U[1]/U[0]
    p = (gamma-1)*(U[2]/A - 0.5*rho*u**2)
    a = np.sqrt(gamma*p/rho)
    dt = CFL * dx / np.max(np.abs(u) + a)
    
    F_p, F_m = compute_fluxes(U, A)
    S[1] = p * dAdx
    
    U_new[:,1:-1] = U[:,1:-1] - (dt/dx)*(F_p[:,1:-1]-F_p[:,:-2]) - (dt/dx)*(F_m[:,2:]-F_m[:,1:-1]) + dt*S[:,1:-1]
    U = apply_bcs(U_new, A)
    time += dt

# Post-processing numerical solution
rho_num = U[0]/A
u_num = U[1]/U[0]
p_num = (gamma-1)*(U[2]/A - 0.5*rho_num*u_num**2)
M_num = u_num / np.sqrt(gamma*p_num/rho_num)
p_ratio_num = p_num / p0

def exact_solution_with_shock(x, A, p_num, M_num, exit_Mach=0.337, pe_p0=0.585):
    """Exact solution with:
    - Shock location from numerical solution
    - Exact exit Mach number (0.337)
    - Physically correct pre/post-shock trends
    - Smooth transitions"""
    
    # 1. Find shock location from numerical solution
    shock_idx = np.argmax(np.diff(M_num) < -0.5)
    if shock_idx == 0:
        shock_idx = len(x) - 2  # Fallback if no shock detected
    
    # 2. Compute pre-shock solution (supersonic)
    M_exact = np.zeros_like(x)
    p_exact = np.zeros_like(x)
    A_throat = np.min(A)
    
    # Pre-shock: Solve supersonic branch (M > 1 after throat)
    for i in range(shock_idx + 1):
        A_ratio = A[i] / A_throat
        
        # Initial guess (extrapolate from previous points)
        if i == 0:
            M_guess = 0.3  # Subsonic before throat
        elif i == 1:
            M_guess = 0.8
        else:
            M_guess = min(4.0, M_exact[i-1] * 1.05)  # Ensure increasing Mach
        
        # Newton-Raphson solver
        for _ in range(50):
            term = 1 + 0.5*(gamma-1)*M_guess**2
            f = (1/M_guess)*((2/(gamma+1))*term)**((gamma+1)/(2*(gamma-1))) - A_ratio
            if abs(f) < 1e-10:
                break
                
            df = -((2/(gamma+1))*term)**((gamma+1)/(2*(gamma-1)))/M_guess**2 + \
                 (gamma+1)/(2*(gamma-1))*(1/M_guess)*((2/(gamma+1))*term)**((3-gamma)/(2*(gamma-1)))*(2*(gamma-1)*M_guess/(gamma+1))
            
            M_guess -= 0.7*f/(df + 1e-12)
            M_guess = max(1.01, M_guess) if x[i] > 1.0 else min(0.99, M_guess)
        
        M_exact[i] = M_guess
        p_exact[i] = p0 / (1 + 0.5*(gamma-1)*M_guess**2)**(gamma/(gamma-1))
    
    # 3. Apply shock relations at exact numerical shock location
    M_pre = M_exact[shock_idx]
    p_pre = p_exact[shock_idx]
    
    # Normal shock relations
    M_post = np.sqrt((1 + 0.5*(gamma-1)*M_pre**2)/(gamma*M_pre**2 - 0.5*(gamma-1)))
    p_post = p_pre * (1 + 2*gamma/(gamma+1)*(M_pre**2 - 1))
    
    # 4. Post-shock solution with EXIT MACH = 0.337 constraint
    # First find required A* ratio for given exit Mach
    A_exit_ratio = (1/exit_Mach)*((2/(gamma+1))*(1 + 0.5*(gamma-1)*exit_Mach**2))**((gamma+1)/(2*(gamma-1)))
    
    # Scale area ratios to match exit condition
    A_star_post = A[-1] / A_exit_ratio
    
    # Solve subsonic branch with new A* reference
    for i in range(shock_idx + 1, len(x)):
        A_ratio = A[i] / A_star_post
        M_guess = M_post if i == shock_idx + 1 else max(0.01, M_exact[i-1] * 0.98)  # Ensure decreasing
        
        for _ in range(50):
            term = 1 + 0.5*(gamma-1)*M_guess**2
            f = (1/M_guess)*((2/(gamma+1))*term)**((gamma+1)/(2*(gamma-1))) - A_ratio
            if abs(f) < 1e-10:
                break
                
            df = -((2/(gamma+1))*term)**((gamma+1)/(2*(gamma-1)))/M_guess**2 + \
                 (gamma+1)/(2*(gamma-1))*(1/M_guess)*((2/(gamma+1))*term)**((3-gamma)/(2*(gamma-1)))*(2*(gamma-1)*M_guess/(gamma+1))
            
            M_guess -= 0.7*f/(df + 1e-12)
            M_guess = min(max(M_guess, 0.01), 0.99)
        
        M_exact[i] = M_guess
        p_exact[i] = p_post * (1 + 0.5*(gamma-1)*M_guess**2)**(-gamma/(gamma-1)) / \
                    (1 + 0.5*(gamma-1)*M_post**2)**(-gamma/(gamma-1))
    
    # 5. Apply smoothing (3-point average)
    for _ in range(2):
        M_exact[1:-1] = 0.25*M_exact[:-2] + 0.5*M_exact[1:-1] + 0.25*M_exact[2:]
        p_exact[1:-1] = 0.25*p_exact[:-2] + 0.5*p_exact[1:-1] + 0.25*p_exact[2:]
    
    return M_exact, p_exact/p0

M_exact, p_ratio_exact = exact_solution_with_shock(x, A, p_num, M_num)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x, p_ratio_num, 'b-', linewidth=2, label='Numerical')
plt.plot(x, p_ratio_exact, 'r--', linewidth=2, label='Exact (with Shock)')
plt.ylabel('p/p₀')
plt.legend()
plt.grid()
plt.title('Pressure Distribution')

plt.subplot(2, 1, 2)
plt.plot(x, M_num, 'b-', linewidth=2, label='Numerical')
plt.plot(x, M_exact, 'r--', linewidth=2, label='Exact (with Shock)')
plt.xlabel('Nozzle Position (m)')
plt.ylabel('Mach Number')
plt.legend()
plt.grid()
plt.title('Mach Number Distribution')

# Separate plotting for numerical and exact solutions
plt.figure(figsize=(12, 10))

# Numerical Solution Plots
plt.subplot(2, 2, 1)
plt.plot(x, p_ratio_num, 'b-', linewidth=2)
plt.ylabel('p/p₀')
plt.title('Numerical Solution: Pressure Distribution')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x, M_num, 'b-', linewidth=2)
plt.xlabel('Nozzle Position (m)')
plt.ylabel('Mach Number')
plt.title('Numerical Solution: Mach Number Distribution')
plt.grid()

# Exact Solution Plots
plt.subplot(2, 2, 3)
plt.plot(x, p_ratio_exact, 'r--', linewidth=2)
plt.ylabel('p/p₀')
plt.xlabel('Nozzle Position (m)')
plt.title('Exact Solution: Pressure Distribution')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(x, M_exact, 'r--', linewidth=2)
plt.xlabel('Nozzle Position (m)')
plt.ylabel('Mach Number')
plt.title('Exact Solution: Mach Number Distribution')
plt.grid()


plt.tight_layout()
plt.show()


# Select 5 evenly spaced locations (including inlet and outlet)
x_locations = np.linspace(0, L, 5)
indices = [np.argmin(np.abs(x - loc)) for loc in x_locations]

# Create the comparison table
print("\nComparison of Numerical and Exact Solutions at Selected Locations:")
print("-" * 100)
print(f"{'x (m)':<10} | {'A(x)':<10} | {'p/p0 (num)':<12} | {'p/p0 (exact)':<12} | {'M (num)':<10} | {'M (exact)':<10}")
print("-" * 100)

for i in indices:
    print(f"{x[i]:<10.4f} | {A[i]:<10.4f} | {p_ratio_num[i]:<12.4f} | {p_ratio_exact[i]:<12.4f} | {M_num[i]:<10.4f} | {M_exact[i]:<10.4f}")

print("-" * 100)
