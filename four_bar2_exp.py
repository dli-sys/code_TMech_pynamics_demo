# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""
import sys
import os

script_dir=os.path.abspath(os.curdir)
local_path = os.path.join(script_dir,'./code_pynamics/python')
local_path = os.path.normpath(os.path.abspath(local_path))
sys.path.insert(0,local_path)


import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration
from pynamics.constraint import AccelerationConstraint,KinematicConstraint
from scipy.optimize import minimize
import sympy
import numpy
import matplotlib.pyplot as plt
from math import pi
from decimal import Decimal

plt.close('all')
plt.ion()



system = System()
pynamics.set_system(__name__,system)


para = []


lA = Constant(0.04,'lA',system)
lh = Constant(0.003,'lh',system)
lT = Constant(0.06,'lT',system)

m = Constant(1,'m',system)

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)
tol = 1e-5
tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]


T1 = sympy.Symbol('T1')
T2 = sympy.Symbol('T2')
T3 = sympy.Symbol('T3')
T4 = sympy.Symbol('T4')

# Fconst = sympy.Symbol('Fconst')

Fx_tip = sympy.Symbol('Fx_tip')
Fy_tip= sympy.Symbol('Fy_tip')
T_tip= sympy.Symbol('T_tip')


qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
qD,qD_d,qD_dd = Differentiable('qD',system)

qE,qE_d,qE_dd = Differentiable('qE',system)

initialvalues = {}
angle_value = 45
angle_tilt = -10
initialvalues[qA]   =(angle_value+angle_tilt)*pi/180
initialvalues[qA_d] =0*pi/180
initialvalues[qB]   =pi-2*(angle_value)*pi/180
initialvalues[qB_d] =0*pi/180
initialvalues[qC]   =pi - (angle_value-angle_tilt)*pi/180
initialvalues[qC_d] =0*pi/180
initialvalues[qD]   =2*angle_value*pi/180 -pi
initialvalues[qD_d] =0*pi/180

initialvalues[qE]   = 0*pi/180
initialvalues[qE_d] = 0
statevariables = system.get_state_variables()

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)
D = Frame('D',system)

E = Frame('E',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(A,[0,0,1],qB,system)
C.rotate_fixed_axis(N,[0,0,1],qC,system)
D.rotate_fixed_axis(B,[0,0,1],qD,system)


E.rotate_fixed_axis(N,[0,0,1],qE,system)

pNA=0*N.x
pAB=pNA+lA*A.x
pBD = pAB + lA*B.x

pNC=pNA
pCD = pNC+lA*C.x
pDB = pCD + lA*D.x

pNE = pNA - lh*E.y

pER = pNE + 0.5*lT*E.x
pEL = pNE - 0.5*lT*E.x


vCD_AB = pAB-pCD
uCD_AB = 1/(vCD_AB.length()) * vCD_AB

vCD=pCD.time_derivative()
vAB=pAB.time_derivative()


statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]

eq = []
eq.append((pBD-pDB).dot(N.x))
eq.append((pBD-pDB).dot(N.y))
eq_d=[(system.derivative(item)) for item in eq]
eq_dd=[(system.derivative(item)) for item in eq_d]

c=KinematicConstraint(eq)

variables = [qA,qB,qC,qD]
guess = [initialvalues[item] for item in variables]
result = c.solve_numeric(variables,guess,system.constant_values)

ini = []
for item in system.get_state_variables():
    if item in variables:
        ini.append(result[item])
    else:
        ini.append(initialvalues[item])
        
statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]



pAcm = pNA+lA/2*A.x
pBcm = pAB+lA/2*B.x
pCcm = pNC+lA/2*C.x
pDcm = pCD+lA/2*D.x

vBD = pBD.time_derivative()
vDB = pDB.time_derivative()

wNA = N.get_w_to(A)
wAB = A.get_w_to(B)
wNC = N.get_w_to(C)
wCD = C.get_w_to(D)

wND = N.get_w_to(D)

uBD = 1/(vBD.length())*vBD
uDB = 1/(vDB.length())*vDB

BodyA = Particle(pAcm,m,'ParticleA',system)
BodyB = Particle(pBcm,m,'ParticleB',system)
BodyC = Particle(pCcm,m,'ParticleC',system)
BodyD = Particle(pDcm,m,'ParticleD',system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wNC,wNC)

system.addforce(T2*uCD_AB,vCD)
system.addforce(-T2*uCD_AB,vAB)
system.addforce(-T2*N.y,vAB)
system.addforce(-T1*N.y,vAB)

system.addforce(T3*uCD_AB,vCD)
system.addforce(-T3*uCD_AB,vAB)
system.addforce(-T3*N.y,vCD)
system.addforce(-T4*N.y,vCD)

eq = []
eq.append((pBD-pDB))
eq.append((pCD-pAB))
eq.append((pAB))
eq_d = []
eq_d.extend([item.time_derivative() for item in eq])
eq_d_scalar = []
eq_d_scalar.append(eq_d[0].dot(N.x))
eq_d_scalar.append(eq_d[0].dot(N.y))
eq_dd = [item.time_derivative() for item in eq_d]
eq_dd_scalar = []
eq_dd_scalar.append(eq_dd[0].dot(N.x))
eq_dd_scalar.append(eq_dd[0].dot(N.y))
eq_dd_scalar.append(eq_dd[1].dot(N.x))
eq_dd_scalar.append(eq_dd[2].dot(N.y))
system.add_constraint(AccelerationConstraint(eq_dd_scalar))

f,ma = system.getdynamics()

fm = sympy.Matrix(f)
fm = fm.subs(initialvalues)
fm = fm.subs({qA_d:0,qB_d:0,qC_d:0,qD_d:0})
fm = fm.subs(system.constant_values)


result = sympy.solve(fm[:], [T1,T2,T3,T4])

vx = vBD.dot(N.x)
vy = vBD.dot(N.y)
wND_scalar = wND.dot(N.z)

q_d = sympy.Matrix([qA_d,qB_d,qC_d,qD_d])
q_ind = sympy.Matrix([qA_d,qC_d])
q_dep = sympy.Matrix([qB_d,qD_d])


v = sympy.Matrix([vx,vy,wND_scalar])
J = v.jacobian(q_d)
J_ind = v.jacobian(q_ind)
J_dep = v.jacobian(q_dep)


zero  = J_ind*q_ind+J_dep*q_dep - J*q_d

eq = pBD-pDB
eq_d = eq.time_derivative()
eq_d_scalar = []
eq_d_scalar.append(eq_d.dot(N.x))
eq_d_scalar.append(eq_d.dot(N.y))
eq_d_scalar= sympy.Matrix(eq_d_scalar)

J_constraint = eq_d_scalar.jacobian(q_d)
A_m= eq_d_scalar.jacobian(q_ind)
B_m= eq_d_scalar.jacobian(q_dep)
C_m = -B_m.inv()*A_m

J_new = (J_ind+J_dep*C_m)

f = sympy.Matrix([Fx_tip,Fy_tip,T_tip])
T_ind = J_new.T*f
T_dep = C_m.inv().T*T_ind


l_3 = (pAB-pCD)
l_3_length = (l_3.dot(l_3))**0.5
# l_4_length = (l_4.dot(l_4))**0.5


l_BE_R = pAB - pER
l_BE_L = pCD - pEL
u_L_BE_R = (1/l_BE_R.length())*l_BE_R
u_L_BE_L = (1/l_BE_L.length())*l_BE_L


# with triangle version
pV1_0 = pAB - 1*lA*u_L_BE_R
pV2_0 = pCD - 1*lA*u_L_BE_L
pV3_0 = pAB - 1*lA*u_L_BE_R - l_3_length*u_L_BE_R
pV4_0 = pCD - 1*lA*u_L_BE_L - l_3_length*u_L_BE_L

v_l1 = pV1_0.time_derivative().dot(u_L_BE_R)
v_l2 = pV2_0.time_derivative().dot(u_L_BE_L)
v_l3 = pV3_0.time_derivative().dot(u_L_BE_R)
v_l4 = pV4_0.time_derivative().dot(u_L_BE_L)

v_t = sympy.Matrix([v_l1,v_l2,v_l3,v_l4])

J_t_dep = v_t.jacobian(q_d)
J_t_ind = v_t.jacobian(q_ind)

# f_tip = sympy.Matrix([Fx_tip,Fy_tip])
# f_t = (J_t_ind)*f_tip

f1 = sympy.Symbol('f1')
f2 = sympy.Symbol('f2')
f3 = sympy.Symbol('f3')
f4 = sympy.Symbol('f4')


cond1 = {}
cond1[lA] = 0.04
cond1[lh] = 0.01
cond1[lT] = 0.06
cond1[Fx_tip] = 0
cond1[Fy_tip] = 0
cond1[T_tip] = 0

f_t_sym = sympy.Matrix([f1,f2,f3,f4])
ft1 = (J_t_ind.T)*f_t_sym


def calculate_force_angle(angle,angle_tilt,plot=False,max_fric=100,cond=cond1):
    print(f'current angle: {angle}')
    print(f'external load: {cond1}')
    initialvalues = {}
    angle_value = angle
    initialvalues[qA]   = (angle_value+angle_tilt)*pi/180
    initialvalues[qA_d] = 0*pi/180
    initialvalues[qB]   = pi-2*(angle_value)*pi/180
    initialvalues[qB_d] = 0*pi/180
    initialvalues[qC]   = pi - (angle_value-angle_tilt)*pi/180
    initialvalues[qC_d] = 0*pi/180
    initialvalues[qD]   = 2*angle_value*pi/180 -pi
    initialvalues[qD_d] = 0*pi/180    
    initialvalues[qE]   = 0*pi/180
    initialvalues[qE_d] = 0
    
    if plot==True:
        statevariables = system.get_state_variables()
        ini0 = [initialvalues[item] for item in statevariables]
        
        draw_skeleton(ini0, [pBD,pNA,pNE],linestyle='solid')
        draw_skeleton(ini0, [pDB,pAB,pNA,pNE],linestyle='dashed')
        draw_skeleton(ini0, [pBD,pCD,pNA,pNE],linestyle='solid')
        draw_skeleton(ini0, [pCD,pV4_0],linestyle='solid')
        draw_skeleton(ini0, [pAB,pV3_0],linestyle='solid')
        draw_skeleton(ini0, [pCD,pV2_0],linestyle='dashdot')
        draw_skeleton(ini0, [pAB,pV1_0],linestyle='dashdot')
      
    ft_error = T_ind-ft1
    ft_error_sym = ft_error.subs(initialvalues).subs(cond1)
    ft_error_sym = ft_error_sym.subs({f1:0,f4:0})
    
    from scipy.optimize import minimize
    from scipy.optimize import differential_evolution
    from scipy.optimize import Bounds
    from scipy.optimize import LinearConstraint
    bounds1 = [(-max_fric,0),(-max_fric,0),(-max_fric,0),(-max_fric,0)]
    
    A_eq  =numpy.array (ft_error_sym.jacobian(sympy.Matrix([f1,f2,f3,f4]))).astype(numpy.float64)
    lb1 = -numpy.array(ft_error_sym.subs({f1:0,f2:0,f3:0,f4:0})).astype(numpy.float64)
    ub1 = -numpy.array(ft_error_sym.subs({f1:0,f2:0,f3:0,f4:0})).astype(numpy.float64)
    lb = numpy.transpose(lb1).reshape(2) - 1e-5
    ub = numpy.transpose(ub1).reshape(2) + 1e-5
    con1 = LinearConstraint(A_eq, lb, ub)
    
    # res = dual_annealing(calculate_f_dump,bounds1)
    res = minimize(lambda x:numpy.sum(x**2),[1,1,1,1],bounds=bounds1,constraints=con1,options={'disp':False})
    print(f'tendon force: {res.x}')
    cal1 = (J_t_ind.subs(initialvalues).subs(cond1).T)*sympy.Matrix([res.x]).T
    cal2 = T_ind.subs(initialvalues).subs(cond1)
    # error = cal1-cal2
    # print(f'T_ind from tendon_force {cal1}')
    # print(f' sum of tendon_force T_ind {cal1[0]+cal1[1]}')
    # print(f'T_ind from tip force {cal2}')
    # print(f' sum of tip force T_ind {cal2[0]+cal2[1]}')

    T_ind_sym = ft1.subs(initialvalues).subs(cond1).T

    return [res.x,cal1.T,T_ind_sym,initialvalues]


def draw_skeleton(ini0,points1,linestyle='solid',color=[],displacement=[0,0],amplify=1):
    # points1 = [pGR,pFR,pER,pAB]
    po2 = PointsOutput(points1, constant_values=system.constant_values)
    po2.calc(numpy.array([ini0,ini0]),[0,1])
    ax = po2.plot_time_c(newPlot=False,linestyle=linestyle,color=color,displacement=displacement,amplify=amplify)
    return ax


def plot_one_config(angle_value,angle_tilt,displacement=[0,0],amplify=1):
    initialvalues = {}  
    initialvalues[qA]   =(angle_value+angle_tilt)*pi/180
    initialvalues[qA_d] =0*pi/180
    initialvalues[qB]   =pi-2*(angle_value)*pi/180
    initialvalues[qB_d] =0*pi/180
    initialvalues[qC]   =pi - (angle_value-angle_tilt)*pi/180
    initialvalues[qC_d] =0*pi/180
    initialvalues[qD]   =2*angle_value*pi/180 -pi
    initialvalues[qD_d] =0*pi/180
    
    initialvalues[qE]   = 0*pi/180
    initialvalues[qE_d] = 0     
    statevariables = system.get_state_variables()
    ini0 = [initialvalues[item] for item in statevariables]  
    ax2 = draw_skeleton(ini0, [pNA,pAB,pBD,pCD,pNA],linestyle='solid',color='k',displacement=displacement,amplify=amplify)
    
    ax2 = draw_skeleton(ini0, [pER,pEL],linestyle='--',color='k',displacement=displacement,amplify=amplify)
    ax2 = draw_skeleton(ini0, [pER,pAB],linestyle='--',color=[0.5,0.5,0.5],displacement=displacement,amplify=amplify)
    ax2 = draw_skeleton(ini0, [pEL,pCD],linestyle='-',color='b',displacement=displacement,amplify=amplify)

    return ax2,initialvalues

num = 4
angle1 = 30
angle2 = 75

angle_tilt_plot = numpy.linspace(30, -30,5)
angle_plot = numpy.linspace(120,30,4)

angle_tilt_s1 = numpy.linspace(-30,30,5)
angles1 = numpy.linspace(75, 30,4)
angle_2d1,angle_2d2=numpy.meshgrid(angle_plot,angle_tilt_plot)

t_maxs_2d = []

plt.rcParams["font.family"] = "Times New Roman"
import time
time.sleep(1)

from scipy import interpolate
data_dir = os.path.join(os.getcwd(),'exp_data')
exp_30= numpy.genfromtxt(os.path.join(data_dir,'inner30.csv'),delimiter=',')
exp_30= numpy.genfromtxt(os.path.join(data_dir,'inner30.csv'),delimiter=',')
exp_60 = numpy.genfromtxt(os.path.join(data_dir,'inner60.csv'),delimiter=',')
exp_90 = numpy.genfromtxt(os.path.join(data_dir,'inner90.csv'),delimiter=',')
exp_120 = numpy.genfromtxt(os.path.join(data_dir,'inner120.csv'),delimiter=',')
exp_30_t = -0.03*numpy.average(exp_30,axis=0)
exp_60_t = -0.03*numpy.average(exp_60,axis=0)
exp_90_t = -0.03*numpy.average(exp_90,axis=0)
exp_120_t = -0.03*numpy.average(exp_120,axis=0)
exp_torques = [[exp_30_t],[exp_60_t],[exp_90_t],[exp_120_t]]
exp_torques = numpy.array([exp_30_t,exp_60_t,exp_90_t,exp_120_t])
# t_force_temp = t_forces[:,0]
# t_force_temp_max = numpy.amax(t_forces,axis=0)
# t_force_temp_min = numpy.amin(t_forces,axis=0)
# exp_angles = numpy.linspace(angle_start,angle_end,7)
# ft_max = interpolate.interp1d(exp_angles,t_force_temp_max,fill_value = 'extrapolate', kind='quadratic')
# ft_min = interpolate.interp1d(exp_angles,t_force_temp_min,fill_value = 'extrapolate', kind='quadratic')

#generate Fig.9(a)

for item in range(0,5):
    t_maxs = []
    for item1 in range(0,4):
        # t_max = (item1**2-item**3)/10000 +10*random.random()
        #Draw config
        x_dis = angle_tilt_plot[item]
        y_dis = angle_plot[item1]
        
        q1_value = angles1[item1]
        ori_value = angle_tilt_s1[item]
        ax1,initialvalues1 = plot_one_config(q1_value,ori_value,[x_dis,y_dis],amplify=100)
        #calcualte max torque
        T_ind_sym = ft1.subs(initialvalues1).subs(cond1).T
        T_s = T_ind_sym.subs({f1:0,f4:0})
        T1 = T_s[0]+T_s[1]
        max_fric  = 1.56
        bounds1 = [(-max_fric,0),(-max_fric,0)]
        t_max = minimize(lambda x:T1.subs({f2:x[0],f3:x[1]}),[0,0],bounds=bounds1)
        # t_max = T1.subs({f2:0,f3:-1.56})
        # print(f'tendon force: {t_max.x}')
        t_max=  float(t_max.fun)
        t_maxs = numpy.append(t_maxs,t_max)  
        
        #add text
        t_max_exp = exp_torques[item1,item]
        rmse_temp = (t_max-t_max_exp)**2
        if (item==4 and item1==3) or (item==0 and item1==3):
            error_string = "%.4f" % (t_max)
            rmse_temp=0
        else:   
            error_string = "%.4f" % (t_max) +",\n " +'%.4f' % (t_max_exp)+ ",\n "+ '%.3e' % rmse_temp         
            
        plt.text(x_dis,y_dis-3,error_string,ha='center',va='top')        
        plt.plot(x_dis,y_dis,'o', color='k',markersize=abs(rmse_temp*3e5),alpha=0.5) 
 

    if ori_value == angle_tilt_s1[0]:
        t_maxs_2d = t_maxs
    else:
        t_maxs_2d = numpy.vstack((t_maxs_2d,t_maxs))
  
        

import matplotlib.colors
ax2=plt.subplot(111)  
im = ax2.pcolormesh(angle_2d2,angle_2d1,t_maxs_2d.astype('float'),shading='gouraud',cmap='coolwarm')
plt.colorbar(im)

ax1.set_ylabel("Inner angle $q_{AC}$($^{\circ}$)",fontsize=10)
ax1.set_xlabel("Orientation $q_a$($^{\circ}$)",fontsize=10)
plt.title("(a) Max torque $T_{tip}$($Nm$)",fontsize=10)


ax2.set_xticks([30,15,0,-15,-30])
ax2.set_yticks([120,90,60,30])

ax2.grid('on')
ax2.set_xticklabels(['30','-15','0','15','30'])
ax2.set_yticklabels(['30','60','90','120'])
plt.xlim([-40,40])
ax1.set_aspect('equal')
ax2.set_xlim([-40,40])
ax2.set_xbound([-40,40])
ax2.set_ybound([20,130])
plt.show()


# Generate Fig.9()


plt.figure()
ax3=plt.subplot(111)

ax3.set_ylabel("Max torque (Nm)",fontsize=10)
ax3.set_xlabel("Orientation $q_a$($^{\circ}$)",fontsize=10)
ax3.set_title("(b) Torque change according to orientation for 90$^{\circ}$ inner angle ",fontsize=10)
ax3.set_xticks([30,15,0,-15,-30])
ax3.set_xticklabels(['30','-15','0','15','30'])
ax3.set_yticks(numpy.linspace(-35,-50,4))
ax3.set_yticklabels(numpy.linspace(-35,-50,4)/1000)
# ax.xlim([-40,40])
ax3.set_aspect('equal')
# ax2.set_aspect('equal')
ax3.set_xlim([-40,40])
ax3.set_xbound([-40,40])
ax3.set_ylim([-56,-35])
ax3.set_ybound([-56,-35])
ax3.grid()
from scipy import interpolate
t_forces = exp_90*-0.03*1000
# t_force_temp = t_forces[:,0]
t_force_temp_max = numpy.amax(t_forces,axis=0)
t_force_temp_min = numpy.amin(t_forces,axis=0)
t_force_temp_avg = numpy.average(t_forces,axis=0)
exp_angles = numpy.linspace(30,-30,5)

ft0 = interpolate.interp1d(exp_angles,t_forces,fill_value = 'extrapolate', kind='quadratic')

ft_max = interpolate.interp1d(exp_angles,t_force_temp_max,fill_value = 'extrapolate', kind='quadratic')
ft_min = interpolate.interp1d(exp_angles,t_force_temp_min,fill_value = 'extrapolate', kind='quadratic')

sim_angles = numpy.linspace(30,-30,50)

ax3.fill_between(exp_angles,ft_max(exp_angles),ft_min(exp_angles),color='b',alpha=0.25,label='_nolegend_')
ax3.plot(exp_angles,ft_max(exp_angles),color='b',alpha=0.4,label='_nolegend_')
ax3.plot(exp_angles,ft_min(exp_angles),color='b',alpha=0.4,label='_nolegend_')
# ax3.fill_between(sim_angles,ft_max(numpy.flip(sim_angles))*1000,ft_min(numpy.flip(sim_angles))*1000,color='r',alpha=0.25)

t_max1 = ft0(sim_angles)
ax3.plot(exp_angles, t_maxs_2d[:,2]*1000,'r',label='Simulated')
ax3.plot(exp_angles, t_force_temp_avg,'b-',marker='d',label='Experiments')

for item in range(0,5):
    x_dis = angle_tilt_plot[item]
    y_dis = -65
    q1_value = 45
    ori_value = angle_tilt_s1[item]
    # print("%.0f" %(q1_value))
    # print("%.0f" %(ori_value))
    ax1,initialvalues1 = plot_one_config(q1_value,ori_value,[x_dis,y_dis],amplify=100)

plt.legend()