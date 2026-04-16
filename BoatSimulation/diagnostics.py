import numpy as np
import matplotlib.pyplot as plt


def run(params: dict, arrays: dict) -> None:
    """
    Print sanity-check numbers and open diagnostic plots.
    Called from main.py when DIAGNOSTICS = True.
    plt.show() is deferred to main.py so all figures appear together.
    """
    # ── Unpack ────────────────────────────────────────────────────────────
    rho          = params['rho']
    g            = params['g']
    B            = params['B']
    d            = params['d']
    L            = params['L']
    h_G          = params['h_G']
    Ix           = params['Ix']
    b44          = params['b44']
    Irw          = params['Irw']
    Aw           = params['Aw']
    k            = params['k']
    ww           = params['ww']
    K            = params['K']
    physics_step = params['physics_step']

    xi          = arrays['xi']
    dx          = arrays['dx']
    N_strips    = arrays['N_strips']
    t           = arrays['t']
    dt          = arrays['dt']
    N_t         = arrays['N_t']
    phi_arr     = arrays['phi_arr']
    phi_dot_arr = arrays['phi_dot_arr']

    # ── Build log ─────────────────────────────────────────────────────────
    F_gravity = rho * g * B * d * L   # exact analytical ship weight

    log = {key: np.zeros(N_t) for key in (
        't', 'phi', 'phi_dot', 'F_buoy', 'F_gravity', 'F_net_vert',
        'x_B', 'z_B', 'x_G', 'z_G', 'GZ',
        'tau_buoy', 'tau_damp', 'tau_rw', 'tau_net', 'eta0',
    )}

    for i in range(N_t):
        ti     = t[i]
        phi_i  = phi_arr[i]
        phid_i = phi_dot_arr[i]

        eta      = Aw * np.cos(k*xi - ww*ti)
        z_bottom = xi*np.sin(phi_i) + (-d)*np.cos(phi_i)
        h_sub    = np.maximum(0.0, eta - z_bottom)
        dF       = rho * g * h_sub * L * dx
        F_buoy   = np.sum(dF)

        if F_buoy > 0:
            x_B = np.sum(xi * dF) / F_buoy
            z_B = np.sum((z_bottom + h_sub/2) * dF) / F_buoy
        else:
            x_B, z_B = 0.0, 0.0

        z_G_body = h_G - d
        x_G      = -z_G_body * np.sin(phi_i)
        z_G      =  z_G_body * np.cos(phi_i)

        GZ       = x_B - x_G
        tau_buoy = F_buoy * GZ
        tau_damp = -b44 * phid_i
        tau_rw   = -Irw * (K[0]*phi_i + K[1]*phid_i)
        tau_net  = tau_buoy + tau_damp + tau_rw

        log['t'][i]          = ti
        log['phi'][i]        = phi_i
        log['phi_dot'][i]    = phid_i
        log['F_buoy'][i]     = F_buoy
        log['F_gravity'][i]  = F_gravity
        log['F_net_vert'][i] = F_buoy - F_gravity
        log['x_B'][i]        = x_B
        log['z_B'][i]        = z_B
        log['x_G'][i]        = x_G
        log['z_G'][i]        = z_G
        log['GZ'][i]         = GZ
        log['tau_buoy'][i]   = tau_buoy
        log['tau_damp'][i]   = tau_damp
        log['tau_rw'][i]     = tau_rw
        log['tau_net'][i]    = tau_net
        log['eta0'][i]       = Aw * np.cos(-ww*ti)

    # ── Console snapshot ──────────────────────────────────────────────────
    print("=" * 55)
    print("  INITIAL CONDITIONS (t = 0)")
    print("=" * 55)
    print(f"  phi              = {log['phi'][0]*180/np.pi:.3f} deg")
    print(f"  phi_dot          = {log['phi_dot'][0]:.4f} rad/s")
    print(f"  eta at x=0       = {log['eta0'][0]:.4f} m")
    print()
    print(f"  F_buoy           = {log['F_buoy'][0]/1e6:.4f} MN")
    print(f"  F_gravity        = {F_gravity/1e6:.4f} MN")
    print(f"  F_net_vertical   = {log['F_net_vert'][0]/1e3:.2f} kN")
    print()
    print(f"  x_B              = {log['x_B'][0]:.4f} m")
    print(f"  z_B              = {log['z_B'][0]:.4f} m")
    print(f"  x_G              = {log['x_G'][0]:.4f} m")
    print(f"  z_G              = {log['z_G'][0]:.4f} m")
    print(f"  GZ (moment arm)  = {log['GZ'][0]:.4f} m")
    print()
    print(f"  tau_buoy         = {log['tau_buoy'][0]/1e6:.4f} MN·m")
    print(f"  tau_damp         = {log['tau_damp'][0]:.2f} N·m")
    print(f"  tau_rw           = {log['tau_rw'][0]/1e3:.2f} kN·m")
    print(f"  tau_net          = {log['tau_net'][0]/1e3:.2f} kN·m")
    print("=" * 55)

    # ── Sanity checks ─────────────────────────────────────────────────────
    print()
    print("SANITY CHECKS")
    print("-" * 55)

    # 1. F_buoy == F_gravity at phi=0, Aw=0 (isolate heave balance)
    #    Rolling changes submerged volume → F_buoy ≠ F_gravity at phi≠0.
    #    This is expected (heave DOF not modelled). Verify it's only the tilt.
    z_bot_upright = -d * np.ones(N_strips)          # flat bottom, no tilt
    h_sub_upright = np.maximum(0.0, -z_bot_upright) # calm water (eta=0)
    F_upright     = rho * g * np.sum(h_sub_upright) * L * dx
    print(f"1. F_buoy at phi=0, Aw=0 (should equal F_gravity):")
    print(f"   F_buoy    = {F_upright/1e6:.4f} MN")
    print(f"   F_gravity = {F_gravity/1e6:.4f} MN")
    print(f"   Match (rtol=1e-3): {np.isclose(F_upright, F_gravity, rtol=1e-3)}")
    print(f"   Note: at phi={phi_arr[0]*180/np.pi:.1f} deg "
          f"F_buoy = {log['F_buoy'][0]/1e6:.4f} MN — "
          f"mismatch is tilt-induced volume change (heave not modelled, expected)")

    # 2. x_B at phi=0, Aw=0 should be exactly zero (symmetric hull, flat water)
    dF_upright = rho * g * h_sub_upright * L * dx
    xB_upright = np.sum(xi * dF_upright) / np.sum(dF_upright)
    print(f"\n2. x_B at phi=0, Aw=0 (should be 0 — symmetric hull):")
    print(f"   x_B = {xB_upright:.6f} m  Match: {np.isclose(xB_upright, 0.0, atol=1e-6)}")

    # 3. GZ == 0 at phi=0, Aw=0 (upright boat in calm water)
    GZ_upright = xB_upright - 0.0   # x_G = 0 when phi = 0
    print(f"\n3. GZ at phi=0, Aw=0 (should be 0):")
    print(f"   GZ = {GZ_upright:.6f} m  Match: {np.isclose(GZ_upright, 0.0, atol=1e-6)}")

    # 4. tau_buoy at t=0 should oppose positive initial roll
    print(f"\n4. tau_buoy at t=0 should be restoring (oppose positive phi):")
    print(f"   phi_0    = {phi_arr[0]*180/np.pi:.1f} deg")
    print(f"   tau_buoy = {log['tau_buoy'][0]/1e3:.2f} kN·m")
    print(f"   Restoring: {log['tau_buoy'][0] < 0}")

    # 5. Natural frequency: open-loop vs closed-loop vs theory
    #    Theory (linearized, open-loop):
    nabla        = B * d * L
    BM           = B**2 / (12*d)
    BG           = h_G - d/2
    GM           = BM - BG
    Delta_GM     = rho * g * nabla * GM
    omega_ol_theory = np.sqrt(abs(Delta_GM) / Ix)

    #    Closed-loop adds Irw*k1 to effective stiffness:
    omega_cl_theory = np.sqrt((abs(Delta_GM) + Irw*K[0]) / Ix)

    #    Open-loop simulation: temporarily zero K, re-simulate, restore
    K_saved   = K.copy()
    K[:]      = 0.0
    phi_ol    = np.zeros(N_t)
    phid_ol   = np.zeros(N_t)
    phi_ol[0] = phi_arr[0]
    for i in range(N_t - 1):
        ddphi, *_ = physics_step(phi_ol[i], phid_ol[i], t[i])
        phid_ol[i+1] = phid_ol[i] + dt * ddphi
        phi_ol[i+1]  = phi_ol[i]  + dt * phid_ol[i]
    K[:] = K_saved

    #    FFT of closed-loop (main simulation) and open-loop runs
    freqs      = np.fft.rfftfreq(N_t, dt)
    def _fft_peak(arr):
        fft_mag = np.abs(np.fft.rfft(arr))
        return 2*np.pi * freqs[np.argmax(fft_mag[1:]) + 1]

    omega_cl_fft = _fft_peak(phi_arr)
    omega_ol_fft = _fft_peak(phi_ol)

    print(f"\n5. Natural frequency breakdown:")
    print(f"   Metacentric geometry:")
    print(f"     BM = {BM:.4f} m  BG = {BG:.4f} m  GM = {GM:.4f} m")
    print(f"     Delta_GM = {Delta_GM/1e6:.4f} MN·m/rad")
    print(f"   Open-loop  theory  = {omega_ol_theory:.4f} rad/s  "
          f"(sqrt(Delta_GM / Ix))")
    print(f"   Open-loop  FFT     = {omega_ol_fft:.4f} rad/s")
    print(f"   Closed-loop theory = {omega_cl_theory:.4f} rad/s  "
          f"(sqrt((Delta_GM + Irw*k1) / Ix))")
    print(f"   Closed-loop FFT    = {omega_cl_fft:.4f} rad/s  "
          f"(FFT picks damped ωd ≤ ωn)")
    print(f"   K = {K_saved}")

    print("-" * 55)

    # ── Diagnostic plots ──────────────────────────────────────────────────
    fig_diag, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig_diag.suptitle('Diagnostic Logs', fontsize=14)

    axes[0, 0].plot(log['t'], log['F_buoy']/1e6, 'b', label='F_buoy')
    axes[0, 0].axhline(F_gravity/1e6, color='r', ls='--', label='F_gravity')
    axes[0, 0].set_ylabel('Force (MN)')
    axes[0, 0].set_title('Buoyancy vs Gravity')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(log['t'], log['F_net_vert']/1e3, 'b')
    axes[0, 1].axhline(0, color='k', ls='--')
    axes[0, 1].set_ylabel('Force (kN)')
    axes[0, 1].set_title('Net Vertical Force (Buoyancy − Gravity)')
    axes[0, 1].grid(True)

    axes[1, 0].plot(log['t'], log['x_B'], 'g', label='x_B')
    axes[1, 0].plot(log['t'], log['x_G'], 'r', label='x_G')
    axes[1, 0].set_ylabel('x position (m)')
    axes[1, 0].set_title('Center of Buoyancy vs Center of Gravity')
    axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(log['t'], log['GZ'], 'b')
    axes[1, 1].axhline(0, color='k', ls='--')
    axes[1, 1].set_ylabel('GZ (m)')
    axes[1, 1].set_title('Righting Lever GZ')
    axes[1, 1].grid(True)

    axes[2, 0].plot(log['t'], log['tau_buoy']/1e6, label='tau_buoy (MN·m)')
    axes[2, 0].plot(log['t'], log['tau_damp']/1e3,  label='tau_damp (kN·m)')
    axes[2, 0].plot(log['t'], log['tau_rw']/1e3,    label='tau_rw (kN·m)')
    axes[2, 0].set_ylabel('Torque')
    axes[2, 0].set_title('Individual Torques')
    axes[2, 0].legend(); axes[2, 0].grid(True)

    axes[2, 1].plot(log['t'], log['tau_net']/1e3, 'k', label='closed-loop')
    axes[2, 1].axhline(0, color='r', ls='--')
    axes[2, 1].set_ylabel('Torque (kN·m)')
    axes[2, 1].set_title('Net Torque')
    axes[2, 1].grid(True)

    for ax in axes.flat:
        ax.set_xlabel('Time (s)')

    # Extra figure: open-loop vs closed-loop roll comparison
    fig_freq, (ax_roll, ax_fft) = plt.subplots(1, 2, figsize=(14, 5))
    fig_freq.suptitle('Open-loop vs Closed-loop Frequency Check', fontsize=13)

    ax_roll.plot(t, phi_arr*180/np.pi, 'b', lw=1.2, label='closed-loop (K active)')
    ax_roll.plot(t, phi_ol*180/np.pi,  'r', lw=1.2, label='open-loop  (K = 0)',
                 alpha=0.7)
    ax_roll.axhline(0, color='k', lw=0.8, ls='--')
    ax_roll.set_xlabel('Time (s)'); ax_roll.set_ylabel('Roll (deg)')
    ax_roll.set_title('Roll angle comparison')
    ax_roll.legend(); ax_roll.grid(True)

    freqs_hz  = freqs[1:]
    fft_cl_db = 20*np.log10(np.abs(np.fft.rfft(phi_arr)[1:]) + 1e-12)
    fft_ol_db = 20*np.log10(np.abs(np.fft.rfft(phi_ol)[1:]) + 1e-12)
    ax_fft.plot(freqs_hz * 2*np.pi, fft_cl_db, 'b', lw=1.2,
                label=f'closed-loop  ωd={omega_cl_fft:.2f} rad/s')
    ax_fft.plot(freqs_hz * 2*np.pi, fft_ol_db, 'r', lw=1.2, alpha=0.7,
                label=f'open-loop    ωd={omega_ol_fft:.2f} rad/s')
    ax_fft.axvline(omega_ol_theory,  color='r', ls='--', lw=1,
                   label=f'theory OL  ωn={omega_ol_theory:.2f} rad/s')
    ax_fft.axvline(omega_cl_theory,  color='b', ls='--', lw=1,
                   label=f'theory CL  ωn={omega_cl_theory:.2f} rad/s')
    ax_fft.set_xlabel('ω (rad/s)'); ax_fft.set_ylabel('Magnitude (dB)')
    ax_fft.set_title('FFT of roll (dB)')
    ax_fft.set_xlim(0, 8); ax_fft.legend(fontsize=8); ax_fft.grid(True)

    plt.tight_layout()
    # plt.show() is intentionally omitted — main.py calls it once for all figures
