import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Latin Modern Roman'
matplotlib.rcParams['font.size'] = 16

params = {'text.usetex': False, 'mathtext.fontset': 'cm'} # for the fonts in the legend an inline latex
plt.rcParams.update(params)

def hamilton_verlet(t,y,N,f,dt): # evolve the system according to Newton's law
    x = y[:N]
    p = y[N:]

    # Verlet algorithm (Leapfrog for Newton's equations)
    x = x+dt/2*p
    p = p-dt*(x-f(t))*np.exp(-(x-f(t))**2/2)
    x = x+dt/2*p

    return np.concatenate((x, p))

# simulation parameters
# units: [E]=trap depth, [x]=trap width, [m]=m_Rb. Therefore [t]=1/omega.
hbar  = 1.05E-34
m     = 87*1.66E-27
T     = 2E-6
kB    = 1.38E-23
omega = 3.7*2*np.pi
zR    = 7.3E-3
d     = 400E-3/zR # dimensionless distance
t_f   = 1.1*omega # dimensionless time

# classical trajectory
def f(t):
    return (-d+d*(10*(t/t_f)**3-15*(t/t_f)**4+6*(t/t_f)**5)) if (t < t_f) else 0
# counterdiabatic trajectory
def fcd(t):
    return (-d+d*(10*(t/t_f)**3-15*(t/t_f)**4+6*(t/t_f)**5)+d/t_f**2*(60*(t/t_f)-180*(t/t_f)**2+120*(t/t_f)**3)) if (t < t_f) else 0


# effective temperature in the Wigner function
beta = 2*m*zR**2*omega/hbar*np.tanh(hbar*omega/2/kB/T)

# draw N samples from finite temperature Wigner function
N   = 2000
p0  = randn(N)/np.sqrt(beta)
x0  = randn(N)/np.sqrt(beta)+f(0)
y   = np.concatenate((x0,p0)) # total state vector
ycd = y

# time vector
Nt = 8000
s  = np.linspace(0,10*t_f,Nt)
dt = s[1]-s[0]

x   = np.zeros((Nt, N))
xcd = np.zeros((Nt, N))
p   = np.zeros((Nt, N))
pcd = np.zeros((Nt, N))

for i in range(Nt):
    y = hamilton_verlet(s[i], y, N, f, dt)
    ycd = hamilton_verlet(s[i], ycd, N, fcd, dt)
    # classical
    x[i] = y[:N]
    p[i] = y[N:]
    # counterdiabatic
    xcd[i] = ycd[:N]
    pcd[i] = ycd[N:]

# expected position
xm = np.mean(x, axis=1)
pm = np.mean(p, axis=1)
# expected position (counterdiabatic)
xm_cd = np.mean(xcd, axis=1)
pm_cd = np.mean(pcd, axis=1)

# variance of the position
varx = np.var(x, axis=1)
varp = np.var(p, axis=1)
# variance of the position (counterdiabatic)
varx_cd = np.var(xcd, axis=1)
varp_cd = np.var(pcd, axis=1)

# take the time-averaged variance over the quasi-stationary state and thereby extract the temperature
Tx = beta*np.mean(varx[-1000:])
Tp = beta*np.mean(varp[-1000:])

Txcd = beta*np.mean(varx_cd[-1000:])
Tpcd = beta*np.mean(varp_cd[-1000:])

# In the case of equipartition, temperatures should be equal for x and p
# temperatures are given in terms of the original temperature, i.e. indicate how much the system has heated
# (only make sense in the regime where the mean position is still in the trap - otherwise the entire gas escaped)


# plot figures of merit for a single transport time
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 10))

ax1.plot(s/omega,(xm+d)*zR*100, label='classical')
ax1.plot(s/omega,(xm_cd+d)*zR*100, label='counterdiabatic')
ax1.set_ylabel('position $x$ (cm)')
ax1.legend(loc=4)

ax2.plot(s/omega, pm)
ax2.plot(s/omega, pm_cd)
ax2.set_ylabel('momentum $p$ (a.u.)')

ax3.plot(s/omega,varx/100)
ax3.plot(s/omega,varx_cd/100)
ax3.set_ylabel('variance $\Delta x$ (a.u.)')

ax4.plot(s/omega,varp*100)
ax4.plot(s/omega,varp_cd*100)
ax4.set_ylabel('variance $\Delta p$ (a.u.)')
ax4.set_xlabel('time (s)')

plt.savefig('twa1_runout.pdf')
plt.show()


# plot temperatures for the classical and counterdiabatic trajectories, for several t_f
Nsim = 10
times = np.arange(1*omega, 2.65*omega, 0.015*omega)
temperatures = np.zeros((len(times),Nsim))
temperaturescd = np.zeros((len(times),Nsim))
for t_index,t_f in enumerate(times):
    for n in range(Nsim):
        y = np.concatenate((x0, p0))  # total state vector
        ycd = y
        for i in range(Nt):
            y = hamilton_verlet(s[i], y, N, f, dt)
            ycd = hamilton_verlet(s[i], ycd, N, fcd, dt)
            # classical
            x[i] = y[:N]
            p[i] = y[N:]
            # counterdiabatic
            xcd[i] = ycd[:N]
            pcd[i] = ycd[N:]

        # variance of the position
        varx = np.var(x, axis=1)
        varp = np.var(p, axis=1)
        # variance of the position (counterdiabatic)
        varx_cd = np.var(xcd, axis=1)
        varp_cd = np.var(pcd, axis=1)

        # take the time-averaged variance over the quasi-stationary state and thereby extract the temperature
        Tx = beta * np.mean(varx[-1000:])
        Tp = beta * np.mean(varp[-1000:])

        Txcd = beta * np.mean(varx_cd[-1000:])
        Tpcd = beta * np.mean(varp_cd[-1000:])

        temperatures[t_index, n]   = Tx
        temperaturescd[t_index, n] = Txcd

plt.figure(figsize=(9,5.5))
plt.errorbar(times/omega, np.mean(temperatures, axis=1), yerr=np.sqrt(np.var(temperatures, axis=1)), label='classical', fmt='.-')
plt.errorbar(times/omega, np.mean(temperaturescd, axis=1), yerr=np.sqrt(np.var(temperaturescd, axis=1)), label='counterdiabatic', fmt='.-')
plt.xlabel('transport time $t_f$ (s)')
plt.ylabel('heating $T_2/T_1$')
plt.ylim((0, 6))
plt.margins(0)
plt.axhline(y=1, color='darkgray', linestyle='-')
plt.axvline(x=1.3, color='darkgray', linestyle='-')
plt.legend(loc=1)
plt.savefig('twa2.pdf')
plt.show()
