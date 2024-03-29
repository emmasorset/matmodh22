
# %% Imports and setting line break width for arrays
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl

np.set_printoptions(linewidth=160)

# %% Create the block tridiagonal matrix
def get_AB_dirichlet(nx, ny, mu):
    """Create the Crank-Nicholson discretization matrices.
       Made under the assumption that the rows sum to 1.

    Args:
        nx (int): # points in x direction
        ny (int): # points in x direction
        mu (float): D*dt / (2*nx*ny)

    Returns:
        A, B: LHS and RHS for the CN scheme.
    """
    # Create blocks for the diagonal
    offsets = [-1,0,1]
    A_diag_vals = [-mu, 1+4*mu, -mu]
    B_diag_vals = [mu, 1-4*mu, mu]
    A_diag_block = sp.diags(A_diag_vals, offsets, shape=[nx, nx])  # type: ignore
    B_diag_block = sp.diags(B_diag_vals, offsets, shape=[nx, nx])  # type: ignore
    A_diag_block = sp.csc_matrix(A_diag_block)
    B_diag_block = sp.csc_matrix(B_diag_block)
    # A_diag_block[[0, -1],[0,-1]] = 1+3*mu
    # B_diag_block[[0,-1],[0,-1]] = 1-3*mu
    # ------------ CHANGES -----------------
    A_diag_block[[0, -1],[0,-1]] = 1
    A_diag_block[[0, -1],[1,-2]] = 0
    B_diag_block[[0,-1],[0,-1]] = 1
    B_diag_block[[0, -1],[1,-2]] = 0
    # --------------------------------------

    # Create blocks for upper left and lower right corner
    # A_corner_vals = [-mu, 1+3*mu, -mu]
    # B_corner_vals = [mu, 1-3*mu, mu]
    # A_corner_block = sp.diags(A_corner_vals, offsets, shape=[nx, nx])  # type: ignore
    # B_corner_block = sp.diags(B_corner_vals, offsets, shape=[nx, nx])  # type: ignore
    # ------------ CHANGES -----------------
    A_corner_block = sp.identity(nx)
    B_corner_block = sp.identity(nx)
    # --------------------------------------
    A_corner_block = sp.csc_matrix(A_corner_block)
    B_corner_block = sp.csc_matrix(B_corner_block)
    # A_corner_block[[0,-1],[0,-1]] = 1+2*mu
    # B_corner_block[[0,-1],[0,-1]] = 1-2*mu

    # Create offdiagonal blocks
    A_offdiag_block = sp.diags([-mu], [0], shape=[nx, nx])  # type: ignore
    B_offdiag_block = sp.diags([mu], [0], shape=[nx, nx])  # type: ignore
    # ------------ CHANGES -----------------
    A_offdiag_block = sp.csc_matrix(A_offdiag_block)
    B_offdiag_block = sp.csc_matrix(B_offdiag_block)
    A_offdiag_block[[0,-1],[0,-1]] = 0
    B_offdiag_block[[0, -1],[0,-1]]= 0
    # --------------------------------------


    # Assembly
    A_mosaic = [
        [A_diag_block if i==j else A_offdiag_block if abs(i-j)==1 else None for i in range(ny)]
        for j in range(ny)
    ]
    B_mosaic = [
        [B_diag_block if i==j else B_offdiag_block if abs(i-j)==1 else None for i in range(ny)]
        for j in range(ny)
    ]
    A = sp.csc_matrix(sp.bmat(A_mosaic))
    B = sp.csc_matrix(sp.bmat(B_mosaic))

    # Fix the corner blocks
    A[:nx, :nx] = A_corner_block
    A[-nx:, -nx:] = A_corner_block
    B[:nx, :nx] = B_corner_block
    B[-nx:, -nx:] = B_corner_block
    # ------------ CHANGES -----------------
    A[:nx, nx:2*nx] = 0
    A[-nx:, -2*nx:-nx] = 0
    B[:nx, nx:2*nx] = 0
    B[-nx:, -2*nx:-nx] = 0
    # --------------------------------------

    return A, B

def get_AB_neumann(nx, ny, mu):
    """Create the Crank-Nicholson discretization matrices.
       Made under the assumption that the rows sum to 1.

    Args:
        nx (int): # points in x direction
        ny (int): # points in x direction
        mu (float): D*dt / (2*nx*ny)

    Returns:
        A, B: LHS and RHS for the CN scheme.
    """
    # Create blocks for the diagonal
    offsets = [-1,0,1]
    A_diag_vals = [-mu, 1+4*mu, -mu]
    B_diag_vals = [mu, 1-4*mu, mu]
    A_diag_block = sp.diags(A_diag_vals, offsets, shape=[nx, nx])  # type: ignore
    B_diag_block = sp.diags(B_diag_vals, offsets, shape=[nx, nx])  # type: ignore
    A_diag_block = sp.csc_matrix(A_diag_block)
    B_diag_block = sp.csc_matrix(B_diag_block)
    A_diag_block[[0, -1],[0,-1]] = 1+3*mu
    B_diag_block[[0,-1],[0,-1]] = 1-3*mu
    # ------------ CHANGES -----------------
    # A_diag_block[[0, -1],[0,-1]] = 1
    # A_diag_block[[0, -1],[1,-2]] = 0
    # B_diag_block[[0,-1],[0,-1]] = 1
    # B_diag_block[[0, -1],[1,-2]] = 0
    # --------------------------------------

    # Create blocks for upper left and lower right corner
    A_corner_vals = [-mu, 1+3*mu, -mu]
    B_corner_vals = [mu, 1-3*mu, mu]
    A_corner_block = sp.diags(A_corner_vals, offsets, shape=[nx, nx])  # type: ignore
    B_corner_block = sp.diags(B_corner_vals, offsets, shape=[nx, nx])  # type: ignore
    # ------------ CHANGES -----------------
    # A_corner_block = sp.identity(nx)
    # B_corner_block = sp.identity(nx)
    # --------------------------------------
    A_corner_block = sp.csc_matrix(A_corner_block)
    B_corner_block = sp.csc_matrix(B_corner_block)
    A_corner_block[[0,-1],[0,-1]] = 1+2*mu
    B_corner_block[[0,-1],[0,-1]] = 1-2*mu

    # Create offdiagonal blocks
    A_offdiag_block = sp.diags([-mu], [0], shape=[nx, nx])  # type: ignore
    B_offdiag_block = sp.diags([mu], [0], shape=[nx, nx])  # type: ignore
    # ------------ CHANGES -----------------
    # A_offdiag_block = sp.csc_matrix(A_offdiag_block)
    # B_offdiag_block = sp.csc_matrix(B_offdiag_block)
    # A_offdiag_block[[0,-1],[0,-1]] = 0
    # B_offdiag_block[[0, -1],[0,-1]]= 0
    # --------------------------------------


    # Assembly
    A_mosaic = [
        [A_diag_block if i==j else A_offdiag_block if abs(i-j)==1 else None for i in range(ny)]
        for j in range(ny)
    ]
    B_mosaic = [
        [B_diag_block if i==j else B_offdiag_block if abs(i-j)==1 else None for i in range(ny)]
        for j in range(ny)
    ]
    A = sp.csc_matrix(sp.bmat(A_mosaic))
    B = sp.csc_matrix(sp.bmat(B_mosaic))

    # Fix the corner blocks
    A[:nx, :nx] = A_corner_block
    A[-nx:, -nx:] = A_corner_block
    B[:nx, :nx] = B_corner_block
    B[-nx:, -nx:] = B_corner_block
    # ------------ CHANGES -----------------
    # A[:nx, nx:2*nx] = 0
    # A[-nx:, -2*nx:-nx] = 0
    # B[:nx, nx:2*nx] = 0
    # B[-nx:, -2*nx:-nx] = 0
    # --------------------------------------

    return A, B

# %% Crank Nicholson solver
class EqnConfig:
    """Struct specifying equation parameters.
    """
    def __init__(self, D, f, n0, r0, b0, conds='dirichlet', sigma=None):
        self.D = D
        self.f = f
        self.n0 = n0
        self.r0 = r0
        self.b0 = b0
        if conds == 'dirichlet':
            self.get_AB = get_AB_dirichlet
        elif conds == 'neumann':
            self.get_AB = get_AB_neumann
        elif conds == 'flow':
            if sigma is None:
                raise ValueError('σ must be specified.')
            self.get_AB = lambda nx, ny, mu: get_AB_flow(nx, ny, mu, sigma)
class SolverConfig:
    """Struct specifying solver configuration.
    """
    def __init__(self, Nx, Ny, dx, dy, dt, tsteps):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.tsteps = tsteps
def simulate_2D(eqn, config):
    """Solves the model equations based on configuration objects.

    Args:
        eqn (EqnConfig): Object containing specification of the problem.
        config (SolverConfig): Object containing configuration of the solver.

    Returns:
        n, r, b: Three [tsteps, nx*ny] matrices containing the concentrations
                 of nurotransmitters, free receptors, and bound receptors.
    """
    # System matrices
    dt, tsteps = config.dt, config.tsteps
    Nx, Ny = config.Nx, config.Ny
    dx, dy = config.dx, config.dy
    A, B = eqn.get_AB(Nx, Ny, dt*eqn.D / (2*dx*dy))
    #A, B = makeA2D(Nx, Ny, dt*eqn.D / (2*dx*dy))
    # History matrices (neurotr., free receptor, bound receptor)
    n = np.zeros((tsteps, Nx*Ny))
    r = np.zeros((tsteps, Nx*Ny))
    b = np.zeros((tsteps, Nx*Ny))
    n[0,:], r[0,:], b[0,:] = eqn.n0, eqn.r0, eqn.b0

    # Technically a no-no in numerics
    Ainv = sl.inv(A)
    AinvB = Ainv@B

    for step in range(tsteps-1):
        reaction = (dt*eqn.f(n[step,:], r[step,:], b[step,:]))
        n[step+1,:] = AinvB@n[step,:] + Ainv @ reaction
        r[step+1,:] = r[step,:] + reaction
        b[step+1,:] = b[step,:] - reaction
    return n, r, b

# %% Animates the solution surface
def get_prepped_solution(nx, ny, tsteps, conds='dirichlet'):
    """Computes the solution to all fields for each time step.

    Args:
        nx (int): # points in x direction
        ny (int): # points in y direction
        tsteps (int): # time points in steps

    Returns:
        n, r, b: Concentration of respectively:
                 neurotransmitters, free receptors, bound receptors.
                 Each return value is a matrix of shape [tsteps, nx*ny]
    """
    # Parameters
    radius = 220e-9
    # Big enough to contain the circular post terminal
    edgesize = 3*(2*radius)
    dx = edgesize / nx
    dy = edgesize / ny
    dt = 1e-10 * (31/nx)**2
    #tsteps = 200
    D = 8e-7
    k1 = 4e6
    km1= 0.5
    def f(n,r,b):
        return -k1*n*r + km1*b

    # Free receptors unif. dist., point conc. of neurotr. in middle
    zero = np.zeros(nx*ny)
    #n0 = zero.copy()
    #r0 = zero.copy()
    #b0 = zero.copy()
    n_rec = 1e3 * np.pi*(radius*1e6)**2  # (Receptor density (μm⁻²)) * (membrane area (μm⁻²)) = # of receptors
    n_sites = 5                      # Number of ventricle release sites
    def dist_rec():
        X = np.linspace(0, edgesize, nx)
        Y = np.linspace(0, edgesize, ny)
        X, Y = np.meshgrid(X, Y)
        rec_geom = lambda x, y: (x-edgesize/2)**2 + (y-edgesize/2)**2 <= radius**2
        rec_conc = np.where(rec_geom(X, Y), 1, 0)
        rec_conc = rec_conc * (n_rec/np.count_nonzero(rec_conc))
        return rec_conc.flatten()
    def dist_neur():
        np.random.seed(42)
        ventricle_radius = 20e-9
        ventricle_content = 5000 / n_sites
        X = np.linspace(0, edgesize, nx)
        Y = np.linspace(0, edgesize, ny)
        X, Y = np.meshgrid(X, Y)
        release_pad = edgesize / 8
        release_sites = np.random.uniform(release_pad, edgesize-release_pad, size=[n_sites, 2])   # Centres of ventricle pops
        neur_geom = lambda x, y, c: (x-c[0])**2 + (y-c[1])**2 <= ventricle_radius**2
        neurotr = np.zeros((nx,ny))
        for c in release_sites:
            neur_conc = np.where(neur_geom(X,Y,c), 1, 0)
            while np.count_nonzero(neur_conc) == 0:
                c = np.random.uniform(0, edgesize, size=[2])
                neur_conc = np.where(neur_geom(X,Y,c), 1, 0)
            neur_conc = neur_conc * (ventricle_content/np.count_nonzero(neur_conc))
            neurotr = neurotr + neur_conc
        return neurotr.flatten()
    n0 = dist_neur()
    r0 = dist_rec()
    b0 = zero.copy()
    # half = (nx*ny) // 2
    # n0[half] = 5000

    sigma = None
    if conds == 'flow':
        sigma = dt / (4*dx)
    eqn = EqnConfig(D, f, n0, r0, b0, conds=conds, sigma=sigma)
    solver_config = SolverConfig(nx, ny, dx, dy, dt, tsteps)
    n, r, b = simulate_2D(eqn, solver_config)
    #A, B = get_AB(nx, ny, dt*D / (2*dx*dy))
    #n, r, b, _ = CN2D(A, B, n0, r0, b0, dt, f)
    #print(np.max(n), np.max(r), np.max(b))
    return n, r, b

def animate_2DConserved(field, nx, ny, tsteps):
    """Produce an animation of a fields evolution in time.

    Args:
        field (ndarray): Matrix of shape [tsteps, nx*ny] giving time evolution of a field.
        nx (int): # points in x direction
        ny (int): # points in y direction
        tsteps (int): # time steps
    """
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore
    plot_args = {'rstride': 1, 'cstride': 1, 'cmap': 'viridis',
                 'linewidth': 0, 'antialiased': False, 'alpha': 0.6}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(0, 1, nx)
    Y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, field[0,:].reshape(nx,ny))  # type: ignore
    def update(t):
        ax.clear()
        ax.set_title(f't = {t}/{tsteps}')
        surf = ax.plot_surface(X, Y, field[t,:].reshape((nx, ny)), **plot_args)  # type: ignore
        ax.set_zlim([0, np.max(field[t,:])])  # type: ignore
        return surf,
    anim = animation.FuncAnimation(fig, update, tsteps, interval=100)
    plt.show()
    plt.close()

# %% Visualize all fields in the model
def plot_all_fields(nx, ny, tsteps, conds):
    """Animates all fields in the model.
    """
    # nx = 31
    # ny = 31
    # tsteps = 200
    n, r, b = get_prepped_solution(nx, ny, tsteps, conds)
    animate_2DConserved(n, nx, ny, tsteps)
    animate_2DConserved(r, nx, ny, tsteps)
    animate_2DConserved(b, nx, ny, tsteps)

# plot_all_fields()
# %% Track number of bound receptor to time the excitation transmission
def time_exitation(nx, ny, max_tsteps, conds):
    # Params pulled from get_prepped_solution, spaghetti
    radius = 220e-9
    dx = 2*radius / nx
    dt = dx**2*1e6

    n, r, b = get_prepped_solution(nx, ny, max_tsteps, conds)
    total = r[0,:].sum()
    b_fraction = b.sum(axis=-1) / total
    excitation_tstep = np.argmax(b_fraction >= 0.5)
    excitation_time = excitation_tstep * dt
    plt.plot(b_fraction)
    plt.suptitle('Fraction of bound receptors')
    plt.title(f'Exitation at t = {excitation_time:5.2e}')
    plt.grid()
    plt.show()
# time_exitation(31, 31, 500)
    

# %%
def test1():
    import numpy as np
    import matplotlib.pyplot as plt
    nx, ny = 41, 41
    edgesize = 1.e-6
    radius = edgesize / 6.
    n_rec = 1e3 * (2*radius*1e6)**2
    X = np.linspace(0, edgesize, nx)
    Y = np.linspace(0, edgesize, ny)
    X, Y = np.meshgrid(X, Y)
    rec_site = lambda x, y: (x-edgesize/2)**2 + (y-edgesize/2)**2 <= radius**2
    rec_conc = np.where(rec_site(X, Y), 1, 0)
    rec_conc = rec_conc * (n_rec/np.count_nonzero(rec_conc))
    print(n_rec)
    print(rec_conc.sum())
    print(rec_conc.max())
    plt.imshow(rec_conc)
#test()
# %%
def test2():
    import numpy as np
    import matplotlib.pyplot as plt
    nx, ny = 41, 41
    edgesize = 1.e-6
    radius = edgesize / 6.
    n_sites = 5
    np.random.seed(42)
    ventricle_radius = 20e-9
    ventricle_content = 5000 / n_sites
    X = np.linspace(0, edgesize, nx)
    Y = np.linspace(0, edgesize, ny)
    X, Y = np.meshgrid(X, Y)
    release_sites = np.random.uniform(0, edgesize, size=[n_sites, 2])   # Centres of ventricle pops
    #release_sites = np.random.choice(nx)
    neur_geom = lambda x, y, c: (x-c[0])**2 + (y-c[1])**2 <= ventricle_radius**2
    neurotr = np.zeros((nx,ny))
    for c in release_sites:
        neur_conc = np.where(neur_geom(X,Y,c), 1, 0)
        while np.count_nonzero(neur_conc) == 0:
            c = np.random.uniform(0, edgesize, size=[2])
            neur_conc = np.where(neur_geom(X,Y,c), 1, 0)
        neur_conc = neur_conc * (ventricle_content/np.count_nonzero(neur_conc))
        neurotr = neurotr + neur_conc
    #return neurotr
    print(neurotr.sum())
    print(neurotr.max())
    plt.imshow(neurotr)
#test()

def debug_matrix():
    A, B = get_AB_flow(6, 6, 10, 2)
    with open('matmodh22/matview_flow.txt', 'w') as file:
        np.savetxt(file, A.todense(), fmt='%3.0f')
        file.write('\n\n')
        np.savetxt(file, B.todense(), fmt='%3.0f')
# %% Run stuff

# plot_animation = 0
# plot_time      = 1
# matrix_debug   = 0
# produce_results= 0

# if plot_animation:
#     plot_all_fields(41, 41, 200)
# if plot_time:
#     time_exitation(41, 41, 1000)
# if matrix_debug:
#     debug_matrix()
# if produce_results:
#     plot_evolution_and_time(41, 41, 1000)

# %% Flow shit

def get_AB_flow(nx, ny, mu, sigma):
    """Create the Crank-Nicholson discretization matrices.
       Made using Dirichlet conditions. The velocity field is
       constant and hard coded, since the code is only meant
       for exploration and reporting, not for actual production.
       Also, we assume nx=ny (they will both be passed in for
       compatibility reasons). See the report for a detailed
       description of the matrix structures.

    Args:
        nx (int): # points in x direction
        ny (int): # points in x direction
        mu (float): D*dt / (2*dx*dx)
        sigma (float): dt / (2*dx)

    Returns:
        A, B: LHS and RHS for the CN scheme.
    """
    # Flow vector
    v1 = 5
    v2 = -5
    xflow = sigma * v1
    yflow = sigma * v2
    # Create blocks for the diagonal
    offsets = [-1,0,1]
    A_diag_vals = [-mu-xflow, 1+4*mu, -mu+xflow]
    B_diag_vals = [mu+xflow, 1-4*mu, mu-xflow]
    A_diag_block = sp.diags(A_diag_vals, offsets, shape=[nx, nx])  # type: ignore
    B_diag_block = sp.diags(B_diag_vals, offsets, shape=[nx, nx])  # type: ignore
    A_diag_block = sp.csc_matrix(A_diag_block)
    B_diag_block = sp.csc_matrix(B_diag_block)
    # A_diag_block[[0, -1],[0,-1]] = 1+3*mu
    # B_diag_block[[0,-1],[0,-1]] = 1-3*mu
    # ------------ CHANGES -----------------
    A_diag_block[[0, -1],[0,-1]] = 1
    A_diag_block[[0, -1],[1,-2]] = 0
    B_diag_block[[0,-1],[0,-1]] = 1
    B_diag_block[[0, -1],[1,-2]] = 0
    # --------------------------------------

    # Create blocks for upper left and lower right corner
    # A_corner_vals = [-mu, 1+3*mu, -mu]
    # B_corner_vals = [mu, 1-3*mu, mu]
    # A_corner_block = sp.diags(A_corner_vals, offsets, shape=[nx, nx])  # type: ignore
    # B_corner_block = sp.diags(B_corner_vals, offsets, shape=[nx, nx])  # type: ignore
    # ------------ CHANGES -----------------
    A_corner_block = sp.identity(nx)
    B_corner_block = sp.identity(nx)
    # --------------------------------------
    A_corner_block = sp.csc_matrix(A_corner_block)
    B_corner_block = sp.csc_matrix(B_corner_block)
    # A_corner_block[[0,-1],[0,-1]] = 1+2*mu
    # B_corner_block[[0,-1],[0,-1]] = 1-2*mu

    # Create offdiagonal blocks
    A_left_offdiag_block = sp.diags([-mu+yflow], [0], shape=[nx, nx])  # type: ignore
    A_right_offdiag_block = sp.diags([-mu-yflow], [0], shape=[nx, nx])  # type: ignore
    B_left_offdiag_block = sp.diags([mu-yflow], [0], shape=[nx, nx])  # type: ignore
    B_right_offdiag_block = sp.diags([mu+yflow], [0], shape=[nx, nx])  # type: ignore
    # ------------ CHANGES -----------------
    A_left_offdiag_block = sp.csc_matrix(A_left_offdiag_block)
    A_right_offdiag_block = sp.csc_matrix(A_right_offdiag_block)
    B_left_offdiag_block = sp.csc_matrix(B_left_offdiag_block)
    B_right_offdiag_block = sp.csc_matrix(B_right_offdiag_block)
    A_left_offdiag_block[[0,-1],[0,-1]] = 0
    A_right_offdiag_block[[0,-1],[0,-1]] = 0
    B_left_offdiag_block[[0, -1],[0,-1]]= 0
    B_right_offdiag_block[[0, -1],[0,-1]]= 0
    # --------------------------------------


    # Assembly
    A_mosaic = [
        [A_diag_block if i==j else A_left_offdiag_block if i-j==1 else A_right_offdiag_block if i-j==-1 else None for i in range(ny)]
        for j in range(ny)
    ]
    B_mosaic = [
        [B_diag_block if i==j else B_left_offdiag_block if i-j==1 else B_right_offdiag_block if i-j==-1 else None for i in range(ny)]
        for j in range(ny)
    ]
    A = sp.csc_matrix(sp.bmat(A_mosaic))
    B = sp.csc_matrix(sp.bmat(B_mosaic))

    # Fix the corner blocks
    A[:nx, :nx] = A_corner_block
    A[-nx:, -nx:] = A_corner_block
    B[:nx, :nx] = B_corner_block
    B[-nx:, -nx:] = B_corner_block
    # ------------ CHANGES -----------------
    A[:nx, nx:2*nx] = 0
    A[-nx:, -2*nx:-nx] = 0
    B[:nx, nx:2*nx] = 0
    B[-nx:, -2*nx:-nx] = 0
    # --------------------------------------

    return A, B

# %% Production plots
def produce_evolutions_and_time(nx, ny, tsteps, conds):
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore
    plot_kwargs = {'rstride': 1, 'cstride': 1, 'cmap': 'viridis',
                 'linewidth': 0, 'antialiased': False, 'alpha': 0.6}

    n, r, b = get_prepped_solution(nx, ny, tsteps, conds)
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    mid = tsteps // 2
    dt = 1e-10 * (31/nx)**2     # Ugly, copied from get_prepped_solution in order to avoid refactoring
    # Plot neurotransmitters
    def plot_field(field, name):
        nfig, [nax1, nax2, nax3] = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=[12, 5])
        nfig.suptitle(name, fontweight='bold', fontsize=20)
        nax1.plot_surface(X, Y, field[0,:].reshape((nx,ny)), **plot_kwargs)  # type: ignore
        nax1.set_title(f't = 0 ns')
        nax1.set_xlabel(r'$x$')
        nax1.set_ylabel(r'$y$')

        nax2.plot_surface(X, Y, field[mid,:].reshape((nx,ny)), **plot_kwargs)  # type: ignore
        nax2.set_title(f't = {mid*dt*1e9:.0f} ns')
        nax2.set_xlabel(r'$x$')
        nax2.set_ylabel(r'$y$')

        nax3.plot_surface(X, Y, field[-1,:].reshape((nx,ny)), **plot_kwargs)  # type: ignore
        nax3.set_title(f't = {tsteps*dt*1e9:.0f} ns')
        nax3.set_xlabel(r'$x$')
        nax3.set_ylabel(r'$y$')
        nfig.savefig(f'plots/{conds}/'+name.replace(' ', '_')+f'_{conds}.pdf', format='pdf')

    plot_field(n, 'Neurotransmitters')
    plot_field(r, 'Free receptors')
    plot_field(b, 'Bound receptors')



    total = r[0,:].sum()
    # total = r.sum(axis=-1)
    b_fraction = b.sum(axis=-1) / total
    excitation_tstep = np.argmax(b_fraction >= 0.5)
    excitation_time = excitation_tstep * dt

    plt.style.use('seaborn')
    tfig, tax = plt.subplots(1, 1, figsize=[8, 5])
    tfig.suptitle('Fraction of bound receptors', fontweight='bold', fontsize=20)
    taxis = np.arange(tsteps) * dt * 1e9
    tax.plot(taxis, b_fraction)
    tax.set_title(f'Exitation at t = {excitation_time*1e9:.1f} ns')
    tax.set_xlabel('ns')
    tax.set_ylabel('Bound fraction')
    tfig.savefig(f'plots/{conds}/Bound_fraction_{conds}.pdf', format='pdf')

    # plt.show()
