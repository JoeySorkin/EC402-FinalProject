import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from robot import Robot

# ─────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────
# rho  = 1025.0
# g    = 9.81
# B    = 8.6
# d    = 2.3
# L    = 51.5
# h_G  = 1.24
# Ix   = 2.38e6
# b44  = 6.12e5
# Irw  = 1e5

rho  = 1025.0
g    = 9.81
B    = 8.6
d    = 2.3
L    = 51.5
h_G  = 1.24
Ix   = 2.3763e6
b44  = 5.00e5
Irw  = 1e5

# Heave parameters
m       = rho * B * d * L        # ship mass from displaced volume
k33     = rho * g * B * L        # heave stiffness = rho*g*waterplane area
b33     = 2 * 0.15 * np.sqrt(m * k33)   # heave damping, zeta=0.15

# Wave
Aw   = 0.5
Lw   = 60.0
k    = 2*np.pi/Lw
ww   = np.sqrt(g*k)

# ── Feature flags ──────────────────────────────
HEAVE       = True    # enable/disable heave dynamics
DIAGNOSTICS = False    # enable/disable diagnostic plots
# ───────────────────────────────────────────────

# ─────────────────────────────────────────
#  Hull strips
# ─────────────────────────────────────────
N_strips = 1000
xi       = np.linspace(-B/2, B/2, N_strips)
dx       = xi[1] - xi[0]

hw = B/2
HULL_BX = np.array([-hw, -hw,  hw,  hw])
HULL_BZ = np.array([ 0.4,  -d,  -d,  0.4])

# ─────────────────────────────────────────
#  Physics step
#  z_hull = vertical position of hull waterline (0 if heave disabled)
# ─────────────────────────────────────────


def controller_step(phi, phi_dot=None):
    # Simple PD controller to stabilize roll at zero
    kP = 1;
    kD = 0;
    tau_rw = -Irw * (kP*phi)
    return tau_rw


run = 0
def physics_step(phi, phi_dot, z_hull, z_dot, t_now, tau_rw_cmd=None):
    global run
    run = run + 1
    # Wave surface at each strip
    eta = Aw * np.cos(k*xi - ww*t_now)

    # Hull bottom in world frame
    # z_hull shifts the entire hull up/down with heave
    z_bottom = xi*np.sin(phi) + (-d)*np.cos(phi) + z_hull

    # Submerged height
    h_sub = np.maximum(0.0, eta - z_bottom)
    
    C44 = rho * g * np.sum(h_sub * xi**2 * dx)  # roll added mass
    if run < 5:
        pass
        # print(f'C44 = {C44:.1f} kg·m²')
    # Buoyancy force per strip
    dF     = rho * g * h_sub * L * dx
    F_buoy = np.sum(dF)

    # Center of buoyancy
    if F_buoy > 0:
        x_B = np.sum(xi * dF) / F_buoy
        z_B = np.sum((z_bottom + h_sub/2) * dF) / F_buoy
    else:
        x_B, z_B = 0.0, 0.0

    # Center of gravity in world frame
    z_G_body = h_G - d
    x_G      = -z_G_body * np.sin(phi)
    z_G      =  z_G_body * np.cos(phi) + z_hull   # heave shifts G too

    # Roll torques
    tau_buoy = F_buoy * (x_B - x_G)
    tau_damp = -b44 * phi_dot
    tau_rw   = tau_rw_cmd if tau_rw_cmd is not None else controller_step(phi, phi_dot)
    tau_net  = tau_buoy + tau_damp + tau_rw
    phi_ddot = tau_net / Ix

    # Heave acceleration
    F_gravity  = m * g
    F_net_vert = F_buoy - F_gravity - b33 * z_dot
    z_ddot     = F_net_vert / m

    return phi_ddot, z_ddot, x_B, z_B, x_G, z_G, F_buoy

# ─────────────────────────────────────────
#  Simulate
# ─────────────────────────────────────────
dt  = 0.02
T   = 40.0
t   = np.arange(0, T, dt)
N_t = len(t)

robot = Robot(simulation_timestep=dt)

phi_arr     = np.zeros(N_t)
phi_dot_arr = np.zeros(N_t)
z_arr       = np.zeros(N_t)     # heave position
z_dot_arr   = np.zeros(N_t)     # heave velocity
tau_rw_arr  = np.zeros(N_t)     # reaction-wheel torque applied to roll dynamics

phi_arr[0] = 0    # initial roll

for i in range(N_t - 1):
    robot.external_sensor(phi_arr[i])
    robot_cmd = robot.controller(t[i])
    tau_rw_arr[i] = -Irw * robot_cmd

    phi_ddot, z_ddot, *_ = physics_step(
        phi_arr[i], phi_dot_arr[i],
        z_arr[i], z_dot_arr[i],
        t[i],
        tau_rw_cmd=tau_rw_arr[i]
    )

    # Semi-implicit Euler is more stable for oscillatory dynamics.
    phi_dot_arr[i+1] = phi_dot_arr[i] + dt * phi_ddot
    phi_arr[i+1]     = phi_arr[i]     + dt * phi_dot_arr[i+1]

    if HEAVE:
        z_dot_arr[i+1] = z_dot_arr[i] + dt * z_ddot
        z_arr[i+1]     = z_arr[i]     + dt * z_dot_arr[i+1]
    # if HEAVE=False, z_arr stays zero throughout

tau_rw_arr[-1] = tau_rw_arr[-2]

# ─────────────────────────────────────────
#  Hull geometry helper
# ─────────────────────────────────────────
def get_hull_world(phi, z_hull):
    R   = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi),  np.cos(phi)]])
    pts = R @ np.vstack([HULL_BX, HULL_BZ])
    return pts[0], pts[1] + z_hull

# ─────────────────────────────────────────
#  Figure setup
# ─────────────────────────────────────────
x_wave = np.linspace(-80, 80, 600)

if HEAVE:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_anim, ax_roll, ax_heave = axes
else:
    fig, (ax_anim, ax_roll) = plt.subplots(1, 2, figsize=(14, 6))

fig.patch.set_facecolor([0.85, 0.93, 1.0])

# --- Animation axes ---
ax_anim.set_facecolor([0.53, 0.81, 0.98])
ax_anim.set_xlim(-80, 80)
ax_anim.set_ylim(-8, 8)
ax_anim.axis('off')
title_txt = ax_anim.set_title('', fontsize=12)

wave_fill_ref = [ax_anim.fill_between(
    x_wave, np.zeros_like(x_wave), -15,
    color=[0.15, 0.45, 0.75], zorder=1)]
wave_line, = ax_anim.plot([], [], color=[0.6, 0.85, 1.0], lw=1.5, zorder=2)

hull_patch = Polygon(np.zeros((4, 2)), closed=True,
                     facecolor=[0.3, 0.3, 0.35],
                     edgecolor='k', lw=1.5, zorder=4)
ax_anim.add_patch(hull_patch)

stripe_line, = ax_anim.plot([], [], 'r-', lw=2.5, zorder=5)
B_dot, = ax_anim.plot([], [], 'go', ms=9, zorder=6, label='B')
G_dot, = ax_anim.plot([], [], 'ro', ms=9, zorder=6, label='G')
ax_anim.legend(loc='upper right', fontsize=8)

# --- Roll plot ---
ax_roll.set_xlim(0, T)
ax_roll.set_ylim(-25, 25)
ax_roll.set_xlabel('Time (s)')
ax_roll.set_ylabel('Roll angle (deg)')
ax_roll.set_title('Roll Angle')
ax_roll.axhline(0, color='k', lw=0.8, ls='--')
ax_roll.grid(True, alpha=0.4)
roll_line, = ax_roll.plot([], [], 'b-', lw=1.5)
roll_dot,  = ax_roll.plot([], [], 'ro', ms=6)

# --- Heave plot (only if enabled) ---
if HEAVE:
    ax_heave.set_xlim(0, T)
    ax_heave.set_ylim(-3, 3)
    ax_heave.set_xlabel('Time (s)')
    ax_heave.set_ylabel('Heave (m)')
    ax_heave.set_title('Heave Position')
    ax_heave.axhline(0, color='k', lw=0.8, ls='--')
    ax_heave.grid(True, alpha=0.4)
    heave_line, = ax_heave.plot([], [], 'g-', lw=1.5)
    heave_dot,  = ax_heave.plot([], [], 'ro', ms=6)

    # Also plot wave height at x=0 for reference
    eta0_arr = Aw * np.cos(-ww*t)
    ax_heave.plot(t, eta0_arr, 'b--', lw=1, alpha=0.5, label='wave at x=0')
    ax_heave.legend(fontsize=8)

# ─────────────────────────────────────────
#  Animation update
# ─────────────────────────────────────────
def update(frame):
    i         = frame
    ti        = t[i]
    phi_i     = phi_arr[i]
    phi_dot_i = phi_dot_arr[i]
    z_i       = z_arr[i]
    z_dot_i   = z_dot_arr[i]

    # Wave
    eta = Aw * np.cos(k*x_wave - ww*ti)
    wave_line.set_data(x_wave, eta)
    wave_fill_ref[0].remove()
    wave_fill_ref[0] = ax_anim.fill_between(
        x_wave, eta, -15,
        color=[0.15, 0.45, 0.75], zorder=1)

    # Hull — position comes from heave state (or wave surface if heave off)
    if HEAVE:
        hull_z = z_i
    else:
        hull_z = Aw * np.cos(-ww*ti)   # fixed to wave surface

    hx, hz = get_hull_world(phi_i, hull_z)
    hull_patch.set_xy(np.column_stack([hx, hz]))

    # Waterline stripe
    R  = np.array([[np.cos(phi_i), -np.sin(phi_i)],
                   [np.sin(phi_i),  np.cos(phi_i)]])
    wl = R @ np.array([[-hw*0.95, hw*0.95], [0.0, 0.0]])
    stripe_line.set_data(wl[0], wl[1] + hull_z)

    # B and G markers
    _, _, x_B, z_B, x_G, z_G, _ = physics_step(
        phi_i, phi_dot_i, z_i, z_dot_i, ti, tau_rw_cmd=tau_rw_arr[i]
    )
    B_dot.set_data([x_B], [z_B])
    G_dot.set_data([x_G], [z_G])

    # Roll plot
    roll_line.set_data(t[:i+1], phi_arr[:i+1]*180/np.pi)
    roll_dot.set_data([ti], [phi_i*180/np.pi])

    # Heave plot
    if HEAVE:
        heave_line.set_data(t[:i+1], z_arr[:i+1])
        heave_dot.set_data([ti], [z_i])

    title_txt.set_text(
        f't = {ti:.1f} s  |  roll = {phi_i*180/np.pi:.1f}°'
        + (f'  |  heave = {z_i:.2f} m' if HEAVE else ''))

    artists = [wave_line, hull_patch, stripe_line,
               B_dot, G_dot, roll_line, roll_dot]
    if HEAVE:
        artists += [heave_line, heave_dot]
    return artists

ani = FuncAnimation(fig, update,
                    frames=range(0, N_t, 4),
                    interval=20, blit=False)

# ─────────────────────────────────────────
#  Diagnostics
# ─────────────────────────────────────────
if DIAGNOSTICS:
    fig_d, axes_d = plt.subplots(2, 3, figsize=(16, 8))
    fig_d.suptitle('Diagnostics', fontsize=14)

    F_buoy_log  = np.zeros(N_t)
    tau_net_log = np.zeros(N_t)
    GZ_log      = np.zeros(N_t)
    xB_log      = np.zeros(N_t)
    xG_log      = np.zeros(N_t)

    for i in range(N_t):
        phi_ddot, z_ddot, x_B, z_B, x_G, z_G, F_b = physics_step(
            phi_arr[i], phi_dot_arr[i],
            z_arr[i], z_dot_arr[i], t[i],
            tau_rw_cmd=tau_rw_arr[i])
        F_buoy_log[i]  = F_b
        GZ_log[i]      = x_B - x_G
        xB_log[i]      = x_B
        xG_log[i]      = x_G
        tau_net_log[i] = F_b*(x_B - x_G) - b44*phi_dot_arr[i] \
                         + tau_rw_arr[i]

    F_gravity = m * g

    axes_d[0,0].plot(t, phi_arr*180/np.pi, 'b')
    axes_d[0,0].set_title('Roll angle (deg)'); axes_d[0,0].grid(True)

    axes_d[0,1].plot(t, F_buoy_log/1e6, 'b', label='F_buoy')
    axes_d[0,1].axhline(F_gravity/1e6, color='r', ls='--', label='F_gravity')
    axes_d[0,1].set_title('Buoyancy vs Gravity (MN)')
    axes_d[0,1].legend(); axes_d[0,1].grid(True)

    axes_d[0,2].plot(t, xB_log, 'g', label='x_B')
    axes_d[0,2].plot(t, xG_log, 'r', label='x_G')
    axes_d[0,2].set_title('x_B and x_G (m)')
    axes_d[0,2].legend(); axes_d[0,2].grid(True)

    axes_d[1,0].plot(t, GZ_log, 'b')
    axes_d[1,0].axhline(0, color='k', ls='--')
    axes_d[1,0].set_title('GZ righting lever (m)'); axes_d[1,0].grid(True)

    axes_d[1,1].plot(t, tau_net_log/1e3, 'k')
    axes_d[1,1].axhline(0, color='r', ls='--')
    axes_d[1,1].set_title('Net torque (kN·m)'); axes_d[1,1].grid(True)

    if HEAVE:
        axes_d[1,2].plot(t, z_arr, 'g', label='heave')
        axes_d[1,2].plot(t, Aw*np.cos(-ww*t), 'b--',
                         alpha=0.5, label='wave at x=0')
        axes_d[1,2].set_title('Heave vs wave (m)')
        axes_d[1,2].legend(); axes_d[1,2].grid(True)
    else:
        axes_d[1,2].text(0.5, 0.5, 'Heave disabled',
                         ha='center', va='center',
                         transform=axes_d[1,2].transAxes, fontsize=12)

    for ax in axes_d.flat:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()

plt.tight_layout()
plt.show()