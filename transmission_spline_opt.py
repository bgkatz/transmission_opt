

# Based on: http://labs.casadi.org/OCP

from casadi import *

N = 100 # number of control intervals

m_body = .5 		# mass
j_rotor = 3e-6 		#rotor inertia
l_leg = .32 			# leg length
tau = 1.6 			# motor torque
p_max = 1500 		# max motor power
g = 9.8 			# gravity

opti = Opti() 		# Optimization problem

# ---- decision variables ---------
X = opti.variable(2,N+1) # state trajectory
pos   = X[0,:]
speed = X[1,:]

accel = opti.variable(1, N+1)

U = opti.variable(2,N+1)   # control trajectory (transmission ratio and its derivative)
k = U[0,:]
kd = U[1,:]


W = opti.variable(2, N+1)
theta = W[0,:]
thetad = W[1,:]

T = opti.variable()      # final time

# ---- objective          ---------
opti.minimize(-(speed[-1])) # maximize speed at the end

# ---- dynamic constraints --------
f = lambda x,u: vertcat(x[1], (u[0]*tau + j_rotor*x[1]*u[1] - m_body*g*(u[0]**2))/(j_rotor + m_body*(u[0]**2))) # dx/dt = f(x,u)
fk = lambda x,u: vertcat(u)

dt = T/N # length of a control intervals
for i in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,i],         U[:,i])
   k2 = f(X[:,i]+dt/2*k1, U[:,i])
   k3 = f(X[:,i]+dt/2*k2, U[:,i])
   k4 = f(X[:,i]+dt*k3,   U[:,i])
   x_next = X[:,i] + dt/6*(k1+2*k2+2*k3+k4) 
   #x_next = X[:,i] + dt*f(X[:,i], U[:,i])
   opti.subject_to(X[:,i+1]==x_next) # close the gaps
   opti.subject_to(accel[i] == k1[1])			# acceleration

   k_next = k[i] + (dt)*(kd[i] + kd[i+1])/2			# transmission ratio integration constraint
   opti.subject_to(k[i+1]==k_next) # close the gaps

   theta_next = theta[i] + dt*(thetad[i] + thetad[i+1])/2
   opti.subject_to(theta[i+1] == theta_next)	# angle/angular velocity integration constraint
   opti.subject_to(thetad[i] == speed[i]/k[i])	# linear and angular velocity transmission ratio constraint
   

# ----- bounds -------
opti.subject_to(opti.bounded(0,kd,40)) # transmission ratio derivative bounds (no infinite slopes)
opti.subject_to(opti.bounded(0.001,k,1)) # transmission ratio bounds
opti.subject_to(opti.bounded(0, thetad, 1000))	# maximum motor angular velocity
opti.subject_to(opti.bounded(.01, T, .1))  # reasonable takeoff time limits


# ---- boundary conditions --------
opti.subject_to(pos[0]==0)   # 0 initial position
opti.subject_to(speed[0]==0) # 0 initial velocity
opti.subject_to(pos[-1]==l_leg)  # final position < leg length
opti.subject_to(theta[0] == 0)	# initial rotor angle of zero
opti.subject_to(thetad[-1]<200)	# final rotor speed < 200 rad/s
opti.subject_to(thetad[-1] == speed[-1]/k[-1])	# linear and angular velocity transmission ratio constraint at the end
 
# ---- misc. constraints  ----------
opti.subject_to(T>=0) # Time must be positive
#opti.subject_to(kd>=0)	# transmission slope positive

# ---- initial values for solver ---
opti.set_initial(speed, 1)
opti.set_initial(k, .02)
opti.set_initial(kd, 0)
opti.set_initial(T, .05)

# ---- solve NLP              ------
opti.solver("ipopt") # set numerical backend
sol = opti.solve()   # actual solve

# ---- post-processing        ------

h_max = (.5*sol.value(speed[-1])**2)/g
print(h_max)

import csv

col1 = sol.value(theta)
col2 = sol.value(k)
#numpy.savetxt("profile.csv", [col1, col2], delimiter=",")

from pylab import* 
from scipy import integrate 

figure
()
plot(sol.value(speed),label="speed")
plot(sol.value(pos),label="pos")
legend(loc="upper left")
draw()
#plot(limit(sol.value(pos)),'r--',label="speed limit")
#step(range(N),sol.value(U),'k',label="throttle")


figure()
polar(sol.value(theta), sol.value(k), label="K")
draw()
figure()
plot(sol.value(theta), sol.value(k), label="K")
draw()

figure()
plot(sol.value(thetad),  label="Motor Speed")
plot(sol.value(accel),  label="Acceleration")
draw()

#figure()
#plot(tau*sol.value(thetad))
#plot(m_body*multiply(sol.value(speed), sol.value(accel)))
#draw()

ke_rotor = .5*j_rotor*square(sol.value(thetad))
ke_body = .5*m_body*square(sol.value(speed))
pe_body = m_body*g*sol.value(pos)
e_input = sol.value(dt)*tau*integrate.cumtrapz(sol.value(thetad))

figure()
plot(ke_rotor, label="ke rotor")
plot(ke_body, label="ke body")
plot(pe_body, label="pe body")
plot(e_input, label="input energy")
legend(loc="upper left")
draw()

show()