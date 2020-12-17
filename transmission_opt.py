

# Based on: http://labs.casadi.org/OCP
# Explanation of the dynamics constraint here: 
# http://build-its-inprogress.blogspot.com/2020/06/simple-dynamics-with-variable.html

from casadi import *
from pylab import* 
import csv
import time
t_start = time.time()
print(t_start)

N = 500 # number of control intervals

m = .75 		          # mass            (kg)
j_rotor = 6e-6 		 # rotor inertia   (kg*m^2)
l_leg = .35 			 # leg length      (m)
tau = 1.6 			    # motor torque    (N-m)
w_max = 1200          # max motor speed (rad/s)
g = 9.8 			       # gravity         (m/s^2)

opti = Opti() 		# Optimization problem

###  decision variables   ###
X = opti.variable(2,N+1) # position and velocity trajectory
pos   = X[0,:]
speed = X[1,:]

accel = opti.variable(1, N+1) # separating this out behaved better than including in X

U = opti.variable(2,N+1)   # transmission ratio and its derivative
k = U[0,:]
kd = U[1,:]


W = opti.variable(2, N+1)  # rotor angle and angular velocity
theta = W[0,:]
thetad = W[1,:]

T = opti.variable()      # final time

#### objective  ###
opti.minimize(-(speed[-1])) # maximize speed at takeoff

####  dynamic constraints   ###
f = lambda x,u: vertcat(x[1], (u[0]*tau + j_rotor*x[1]*u[1]/u[0] - m*g*(u[0]**2))/(j_rotor + m*(u[0]**2))) # dx/dt = f(x,u)


dt = T/N # length of a control intervals
for i in range(N): # loop over control intervals
   k1 = f(X[:,i],  U[:,i])
   k2 = f(X[:,i+1], U[:,i+1])
   x_next = X[:,i] + dt*(k1+k2)/2            # trapezoidal integration
   opti.subject_to(X[:,i+1]==x_next)         # dynamics integration constraint
   opti.subject_to(accel[i] == k1[1])			# acceleration

   k_next = k[i] + dt*(kd[i] + kd[i+1])/2	   # transmission ratio integration constraint
   opti.subject_to(k[i+1]==k_next)        

   theta_next = theta[i] + dt*(thetad[i] + thetad[i+1])/2 
   opti.subject_to(theta[i+1] == theta_next)	   # angle/angular velocity integration constraint
   opti.subject_to(thetad[i] == speed[i]/k[i])	# linear and angular velocity transmission ratio constraint
   

### bounds  ###
opti.subject_to(opti.bounded(0,kd,80))             # transmission ratio derivative bounds (no infinite slopes)
opti.subject_to(opti.bounded(0.0015,k,1))           # transmission ratio bounds (meters per radian)
opti.subject_to(opti.bounded(0, thetad, w_max))	   # maximum motor angular velocity
opti.subject_to(opti.bounded(.01, T, .5))          # reasonable takeoff time limits
opti.subject_to(opti.bounded(0, accel, 1200))      # acceleration limits


####  boundary conditions  ###
opti.subject_to(pos[0]==0)       # 0 initial position
opti.subject_to(speed[0]==0)     # 0 initial velocity
opti.subject_to(pos[-1]==l_leg)  # final position = leg length at takeoff
opti.subject_to(theta[0] == 0)	# initial rotor angle of zero
opti.subject_to(thetad[-1]<200)	# final rotor speed < 200 rad/s
opti.subject_to(thetad[-1] == speed[-1]/k[-1])	# linear and angular velocity transmission ratio constraint at the end
 
#### misc. constraints   ###
opti.subject_to(T>=0) # Time must be positive

####  initial values   ###
opti.set_initial(k, .02)
opti.set_initial(kd, 0)
opti.set_initial(T, .2)

### solve  ###
opti.solver("ipopt") # set numerical backend
sol = opti.solve()   # actual solve

###  post-processing   ###

h_max = (.5*sol.value(speed[-1])**2)/g      # jump apex height
print(h_max)


####  save the transmission profile ####
col1 = sol.value(theta)
col2 = sol.value(k)

r = 7
xcol = r*sin(sol.value(theta))
ycol = r*cos(sol.value(theta))
zcol = 1000*sol.value(pos)

#numpy.savetxt("profile.csv", [col1, col2], delimiter=",")
print(time.time()-t_start)
numpy.savetxt('screw_profile.txt', np.c_[xcol, ycol, zcol], delimiter = ' ')



### Do some plotting ###

figure()
plot(sol.value(speed),label="speed")
plot(sol.value(pos),label="pos")
legend(loc="upper left")
draw()



figure()
polar(sol.value(theta), sol.value(k), label="K")
draw()

figure()
plot(sol.value(theta), sol.value(k), label="K")
legend(loc="upper left")
xlabel('Rotor Angle (rad)')
ylabel('Transmission Ratio')
draw()

figure()
plot(sol.value(thetad),  label="Motor Speed (rad/s)")
plot(sol.value(accel),  label="Vertical Acceleration (m/s^2)")
legend(loc="lower left")
draw()


ke_rotor = .5*j_rotor*square(sol.value(thetad))
ke_body = .5*m*square(sol.value(speed))
pe_body = m*g*sol.value(pos)
e_input = tau*sol.value(theta)

figure()
plot(ke_rotor, label="ke rotor")
plot(ke_body, label="ke body")
plot(pe_body, label="pe body")
plot(e_input, label="input energy")
plot(ke_rotor+ke_body+pe_body, label="total energy")
legend(loc="upper left")
ylabel('Energy (J)')
draw()

figure()
plot(sol.value(pos), sol.value(k))


show()