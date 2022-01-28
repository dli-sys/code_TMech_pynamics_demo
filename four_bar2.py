# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

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

import sympy
import numpy
import matplotlib.pyplot as plt
plt.close('all')
plt.ion()

from math import pi
system = System()
pynamics.set_system(__name__,system)

lA = Constant(1,'lA',system)
lh = Constant(1,'lh',system)
lT = Constant(1,'lT',system)
# lA = Constant(1,'lA',system)
# lA = Constant(1,'lA',system)
# lA = Constant(1,'lA',system)

m = Constant(1,'m',system)

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)
tol = 1e-5
tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

# preload1 = Constant(0*pi/180,'preload1',system)
# preload2 = Constant(0*pi/180,'preload2',system)
# preload3 = Constant(0*pi/180,'preload3',system)

# Ixx_A = Constant(1,'Ixx_A',system)
# Iyy_A = Constant(1,'Iyy_A',system)
# Izz_A = Constant(1,'Izz_A',system)
# Ixx_B = Constant(1,'Ixx_B',system)
# Iyy_B = Constant(1,'Iyy_B',system)
# Izz_B = Constant(1,'Izz_B',system)
# Ixx_C = Constant(1,'Ixx_C',system)
# Iyy_C = Constant(1,'Iyy_C',system)
# Izz_C = Constant(1,'Izz_C',system)

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
# qF,qF_d,qF_dd = Differentiable('qF',system)
# qG,qG_d,qG_dd = Differentiable('qG',system)


initialvalues = {}
angle_value = 55
initialvalues[qA]   =(angle_value)*pi/180
initialvalues[qA_d] =0*pi/180
initialvalues[qB]   =pi-2*(angle_value)*pi/180
initialvalues[qB_d] =0*pi/180
initialvalues[qC]   =pi - (angle_value)*pi/180
initialvalues[qC_d] =0*pi/180
initialvalues[qD]   =2*angle_value*pi/180 -pi
initialvalues[qD_d] =0*pi/180

initialvalues[qE]   = 0*pi/180
initialvalues[qE_d] = 0
# initialvalues[qF]   = 10*pi/180
# initialvalues[qF_d] = 0
# initialvalues[qG]   = -10*pi/180
# initialvalues[qG_d] = 0
# initialvalues[qA]   =60*pi/180
# initialvalues[qA_d] =0*pi/180
# initialvalues[qB]   =60*pi/180
# initialvalues[qB_d] =0*pi/180
# initialvalues[qC]   =120*pi/180
# initialvalues[qC_d] =0*pi/180
# initialvalues[qD]   =-60*pi/180
# initialvalues[qD_d] =0*pi/180

statevariables = system.get_state_variables()

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)
D = Frame('D',system)

E = Frame('E',system)
# F = Frame('F',system)
# G = Frame('G',system)

# V1 = Frame('V_1',system)
# V2 = Frame('V_2',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(A,[0,0,1],qB,system)
C.rotate_fixed_axis(N,[0,0,1],qC,system)
D.rotate_fixed_axis(B,[0,0,1],qD,system)


E.rotate_fixed_axis(N,[0,0,1],qE,system)
# F.rotate_fixed_axis(E,[0,0,1],qF,system)
# G.rotate_fixed_axis(F,[0,0,1],qG,system)

pNA=0*N.x
pAB=pNA+lA*A.x
pBD = pAB + lA*B.x

pNC=pNA
pCD = pNC+lA*C.x
pDB = pCD + lA*D.x

pNE = pNA - lh*E.y
# pEF = pNE +lA*F.y
# pFG = pEF +lA*G.y

pER = pNE + 0.5*lT*E.x
pEL = pNE - 0.5*lT*E.x


# pFR = pEF - lA*F.x
# pFL = pEF + lA*F.x

# pGR = pFG - lA*G.x
# pGL = pFG + lA*G.x


vCD_AB = pAB-pCD
uCD_AB = 1/(vCD_AB.length()) * vCD_AB

vCD=pCD.time_derivative()
vAB=pAB.time_derivative()


# points = [pDB,pCD,pNC,pER,pEL,pNA,pNE,pFR,pFL,pNE,pEF,pGL,pGR,pEF,pNE,pNA,pAB,pBD]
# points = [pDB,pCD,pNC,pER,pEL,pNA,pNE,pNA,pAB,pBD]
points = [pDB,pCD,pNC,pER,pNA,pNE,pNA,pAB,pBD]
# points = [pDB,pCD,pNC,pNA,pAB,pBD]


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
        
po1 = PointsOutput(points, constant_values=system.constant_values)
po1.calc(numpy.array([ini0,ini]),[0,1])
po1.plot_time()

# po1.calc(numpy.array([ini,ini]),[0,1])
# po1.plot_time()


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

wNB = N.get_w_to(B)

uBD = 1/(vBD.length())*vBD
uDB = 1/(vDB.length())*vDB

# IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
# IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
# IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

# BodyA = Body('BodyA',A,pAcm,mA,IA,system)
# BodyB = Body('BodyB',B,pBcm,mB,IB,system)
# #BodyC = Body('BodyC',C,pCcm,mC,IC,system)
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
# dyn = sympy.Matrix(f)-sympy.Matrix(ma)
# eq_dd = sympy.Matrix(eq_dd)


fm = sympy.Matrix(f)
fm = fm.subs(initialvalues)
fm = fm.subs({qA_d:0,qB_d:0,qC_d:0,qD_d:0})
fm = fm.subs(system.constant_values)


result = sympy.solve(fm[:], [T1,T2,T3,T4])

vx = vBD.dot(N.x)
vy = vBD.dot(N.y)
wNB_scalar = wNB.dot(N.z)

q_d = sympy.Matrix([qA_d,qB_d,qC_d,qD_d])
q_ind = sympy.Matrix([qA_d,qC_d])
q_dep = sympy.Matrix([qB_d,qD_d])


v = sympy.Matrix([vx,vy,wNB_scalar])
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
# l_4 = (pCD-pAB)

l_3_length = (l_3.dot(l_3))**0.5
# l_4_length = (l_4.dot(l_4))**0.5


l_BE_R = pAB - pER
l_BE_L = pCD - pEL
# l_EF_R = pER - pFR
# l_EF_L = pEL - pFL
# l_FG_R = pFR - pGR
# l_FG_L = pFL - pGL

# l_BE_R_length = (l_BE_R.dot(l_BE_R))**0.5
# l_BE_L_length = (l_BE_L.dot(l_BE_L))**0.5
# # l_EF_R_length = (l_EF_R.dot(l_EF_R))**0.5
# # l_EF_L_length = (l_EF_L.dot(l_EF_L))**0.5
# # l_FG_R_length = (l_FG_R.dot(l_FG_R))**0.5
# # l_FG_L_length = (l_FG_L.dot(l_FG_L))**0.5

u_L_BE_R = (1/l_BE_R.length())*l_BE_R
u_L_BE_L = (1/l_BE_L.length())*l_BE_L

# Vertical version
# pV1_0 = pAB - 5*lA*N.y
# pV2_0 = pCD - 5*lA*N.y
# pV3_0 = pAB - 5*lA*N.y + l_3_length*N.y
# pV4_0 = pCD - 5*lA*N.y + l_3_length*N.y

# v_l1 = pV1_0.time_derivative().dot(N.y)
# v_l2 = pV2_0.time_derivative().dot(N.y)
# v_l3 = pV3_0.time_derivative().dot(N.y)
# v_l4 = pV4_0.time_derivative().dot(N.y)

# with triangle version
pV1_0 = pAB - 5*lA*u_L_BE_R
pV2_0 = pCD - 5*lA*u_L_BE_L
pV3_0 = pAB - 5*lA*u_L_BE_L + l_3_length*u_L_BE_L
pV4_0 = pCD - 5*lA*u_L_BE_R + l_3_length*u_L_BE_R

v_l1 = pV1_0.time_derivative().dot(u_L_BE_R)
v_l2 = pV2_0.time_derivative().dot(u_L_BE_L)
v_l3 = pV3_0.time_derivative().dot(u_L_BE_L)
v_l4 = pV4_0.time_derivative().dot(u_L_BE_R)

v_t = sympy.Matrix([v_l1,v_l2,v_l3,v_l4])

J_t  = v_t.jacobian(q_d)
J_t_ind = v_t.jacobian(q_ind)

f_tip = sympy.Matrix([Fx_tip,Fy_tip])

f_t = (J_t_ind)*f_tip

f1 = sympy.Symbol('f1')
f2 = sympy.Symbol('f2')
f3 = sympy.Symbol('f3')
f4 = sympy.Symbol('f4')


cond1 = {}
cond1[lA] = 0.05
cond1[lh] = 0.05
cond1[lT] = 0.05
cond1[Fx_tip] = 0
cond1[Fy_tip] = 10
cond1[T_tip] = 0

def calculate_f_dump(x1):
    # cond2 = {}
    # cond2[f1]=x1[0]
    # cond2[f2]=x1[1] 
    # cond2[f3]=x1[2]    
    # cond2[f4]=x1[3]        
    # value1 = ft_error_sym.subs(cond2)    
    # value1 = numpy.array(value1)
    # value2 = numpy.sum(value1**2)*0
    value3 = numpy.sum(numpy.asanyarray(x1)**2)*1
    # value4 = numpy.sum((abs(x1)+x1)**2)*0
    # return value2+value3+value4
    # print(value2)
    return value3


def calculate_force_angle(angle):
    initialvalues = {}
    angle_value = angle
    initialvalues[qA]   =(angle_value)*pi/180
    initialvalues[qA_d] =0*pi/180
    initialvalues[qB]   =pi-2*(angle_value)*pi/180
    initialvalues[qB_d] =0*pi/180
    initialvalues[qC]   =pi - (angle_value)*pi/180
    initialvalues[qC_d] =0*pi/180
    initialvalues[qD]   =2*angle_value*pi/180 -pi
    initialvalues[qD_d] =0*pi/180    
    initialvalues[qE]   = 0*pi/180
    initialvalues[qE_d] = 0
    
    f_t_sym = sympy.Matrix([f1,f2,f3,f4])
    
    ft1 = (J_t_ind.T)*f_t_sym
    ft_error = T_ind-ft1
    ft_error_sym = ft_error.subs(initialvalues).subs(cond1)
    
    
    from scipy.optimize import minimize
    from scipy.optimize import differential_evolution
    from scipy.optimize import Bounds
    from scipy.optimize import LinearConstraint
       
    # calculate_f_dump([0.001,0.001,0.11,0.001])
    
    # bounds1 = [(-1e-5,1e4),(-1e-5,1e4),(-1e-5,1e4),(-1e-5,1e4)]
    # bounds1 = [(-1e3,1e3),(-1e3,1e3),(-1e3,1e3),(-1e3,1e3)]
    bounds1 = [(0,1e3),(0,1e3),(0,1e3),(0,1e3)]
    
    A_eq  =numpy.array ( ft_error_sym.jacobian(sympy.Matrix([f1,f2,f3,f4]))).astype(numpy.float64)
    lb1 = -numpy.array(ft_error_sym.subs({f1:0,f2:0,f3:0,f4:0})).astype(numpy.float64)
    ub1 = -numpy.array(ft_error_sym.subs({f1:0,f2:0,f3:0,f4:0})).astype(numpy.float64)
    lb = numpy.transpose(lb1).reshape(2) - 1e-4
    ub = numpy.transpose(ub1).reshape(2) + 1e-4
    con1 = LinearConstraint(A_eq, lb, ub)
    
    # res = dual_annealing(calculate_f_dump,bounds1)
    res = minimize(calculate_f_dump,[1,1,-1,-1],bounds=bounds1,constraints=con1,method='SLSQP',options={'disp':False})
    
    cal1 = (J_t_ind.subs(initialvalues).subs(cond1).T).dot(res.x)
    cal2 = T_ind.subs(initialvalues).subs(cond1)
    # error = cal1-cal2
    print(cal1)
    print(cal2)
    return res.x

# calculate_force_angle(30)

num = 10
for item in numpy.linspace(30,75,num):
    print(item)
    if item ==0:
        value = calculate_force_angle(item)
    # print(calculate_force_angle(item))
    else:
        value = numpy.vstack([value,calculate_force_angle(item)])
    
plt.figure()
plt.plot(numpy.linspace(0,89,num),value)
plt.legend(["f1","f2","f3","f4"])
plt.grid()
plt

# plt.figure()
# plt.plot(numpy.linspace(0,89,num),values[1::,2])
# plt.legend(["f1","f2","f3","f4"])
# plt.grid()


# print((J_t_ind.subs(initialvalues).subs(cond1).T).dot(res.x))
# print(T_ind.subs(initialvalues).subs(cond1))

# print(ft_error_sym)

# a = ft_error_sym.subs({f1:0,f3:0})
# print(sympy.solve(a))

# b = ft_error_sym.subs({f2:0,f4:0})
# print(sympy.solve(b))