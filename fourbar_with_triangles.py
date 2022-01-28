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

from scipy.optimize import minimize
from scipy.optimize import shgo
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint



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
qF,qF_d,qF_dd = Differentiable('qF',system)
qG,qG_d,qG_dd = Differentiable('qG',system)
# qTip,qTip_d,qTip_dd = Differentiable('qF',system)



initialvalues = {}


initialvalues[qG]   = 0*pi/180
initialvalues[qG_d] = 0
initialvalues[qF]   = 0*pi/180
initialvalues[qF_d] = 0
initialvalues[qE]   = 0*pi/180
initialvalues[qE_d] = 0


d_angle_value = 45
r_angle_value = 0
# initialvalues[qA]   = -(r_angle_value+ (d_angle_value/2))*pi/180
# initialvalues[qA_d] = 0*pi/180
# initialvalues[qB]   = ( 90-2*(d_angle_value))*pi/180
# initialvalues[qB_d] = 0*pi/180
# initialvalues[qC]   = (-r_angle_value -d_angle_value/2)*pi/180
# initialvalues[qC_d] = 0*pi/180
# initialvalues[qD]   = pi - 1*d_angle_value*pi/180
# initialvalues[qD_d] = 0*pi/180

initialvalues[qA]   = (90-d_angle_value+r_angle_value)*pi/180
initialvalues[qA_d] = 0*pi/180
initialvalues[qB]   = (2*d_angle_value)*pi/180
initialvalues[qB_d] = 0*pi/180
initialvalues[qC]   = (90+d_angle_value+r_angle_value)*pi/180
initialvalues[qC_d] = 0*pi/180
initialvalues[qD]   = -(2*d_angle_value)*pi/180
initialvalues[qD_d] = 0*pi/180
# initialvalues[qA]   =(angle_value)*pi/180
# initialvalues[qA_d] =0*pi/180
# initialvalues[qB]   =pi-2*(angle_value)*pi/180
# initialvalues[qB_d] =0*pi/180
# initialvalues[qC]   =pi - (angle_value)*pi/180
# initialvalues[qC_d] =0*pi/180
# initialvalues[qD]   =2*angle_value*pi/180 -pi
# initialvalues[qD_d] =0*pi/180

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
F = Frame('F',system)
G = Frame('G',system)

# V1 = Frame('V_1',system)
# V2 = Frame('V_2',system)

system.set_newtonian(N)

G.rotate_fixed_axis(N,[0,0,1],qG,system)
F.rotate_fixed_axis(G,[0,0,1],qF,system)
E.rotate_fixed_axis(F,[0,0,1],qE,system)

A.rotate_fixed_axis(E,[0,0,1],qA,system)
B.rotate_fixed_axis(A,[0,0,1],qB,system)
C.rotate_fixed_axis(E,[0,0,1],qC,system)
D.rotate_fixed_axis(B,[0,0,1],qD,system)


pNG = 0*N.x
pGR = pNG + 0.5*lT*G.x
pGL = pNG - 0.5*lT*G.x

pGF = pNG + lA*G.y
pFR = pGF + 0.5*lT*F.x
pFL = pGF - 0.5*lT*F.x

pFE = pGF + lA*F.y
pER = pFE + 0.5*lT*E.x
pEL = pFE - 0.5*lT*E.x

pEA = pFE + lA*E.y
pAB = pEA+lA*A.x
pBD = pAB + lA*B.x

pEC = pEA
pCD = pEC+lA*C.x
pDB = pCD + lA*D.x

vCD_AB = pAB-pCD
uCD_AB = 1/(vCD_AB.length()) * vCD_AB

vCD=pCD.time_derivative()
vAB=pAB.time_derivative()

# points = [pNG,pGF,pFE,pEA,pAB,pBD,pDB,pCD,pEC,pDB]
# points = [pNG,pGF,pFE,pEA,pAB,pBD]

points = [pDB,pCD,pEC,pAB,pBD,pEA,pER,pEL,pEA,pFE,pFR,pFL,pFE,pGF,pGR,pGL,pGF,pNG]

statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]

eq = []
eq.append((pBD-pDB).dot(E.x))
eq.append((pBD-pDB).dot(E.y))
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
po1.calc(numpy.array([ini0,ini0]),[0,1])
po1.plot_time()

def draw_skeleton(points1,fig):
    # points1 = [pGR,pFR,pER,pAB]
    po2 = PointsOutput(points1, constant_values=system.constant_values)
    po2.calc(numpy.array([ini0,ini]),[0,1])
    po2.plot_time(fig=fig)

fig1 = plt.figure()
draw_skeleton([pGR,pFR,pER,pAB],fig1)
draw_skeleton([pGL,pFL,pEL,pCD],fig1)


pAcm = pEA+lA/2*A.x
pBcm = pAB+lA/2*B.x
pCcm = pEC+lA/2*C.x
pDcm = pCD+lA/2*D.x

pEcm = pFE + lA/2*E.y
pFcm = pGF + lA/2*F.y
pGcm = pNG + lA/2*G.y

vBD = pBD.time_derivative()
vDB = pDB.time_derivative()

wNA = E.get_w_to(A)
wAB = A.get_w_to(B)
wNC = E.get_w_to(C)
wCD = C.get_w_to(D)

wNB = E.get_w_to(B)

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

BodyE = Particle(pEcm,m,'ParticleE',system)
BodyF = Particle(pFcm,m,'ParticleF',system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wNC,wNC)



eq = []
eq.append((pBD-pDB))
eq.append((pCD-pAB))
eq.append((pAB))
eq_d = []
eq_d.extend([item.time_derivative() for item in eq])
# eq_d_scalar = []
# eq_d_scalar.append(eq_d[0].dot(N.x))
# eq_d_scalar.append(eq_d[0].dot(N.y))
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

# T_dep = C_m.inv().T*T_ind

# T_ind_T1 = J_new_T1.T*f


l_3 = (pAB-pCD)
l_4 = (pCD-pAB)

l_3_length = (l_3.dot(l_3))**0.5
l_4_length = (l_4.dot(l_4))**0.5


l_BE_R = pAB - pER
l_BE_L = pCD - pEL

l_EF_R = pER - pFR
l_EF_L = pEL - pFL

l_FG_R = pFR - pGR
l_FG_L = pFL - pGL


# l_BE_R_length = (l_BE_R.dot(l_BE_R))**0.5
# l_BE_L_length = (l_BE_L.dot(l_BE_L))**0.5
# l_EF_R_length = (l_EF_R.dot(l_EF_R))**0.5
# l_EF_L_length = (l_EF_L.dot(l_EF_L))**0.5
# l_FG_R_length = (l_FG_R.dot(l_FG_R))**0.5
# l_FG_L_length = (l_FG_L.dot(l_FG_L))**0.5

u_L_BE_R = (1/l_BE_R.length())*l_BE_R
u_L_BE_L = (1/l_BE_L.length())*l_BE_L

u_L_EF_R = (1/l_EF_R.length())*l_EF_R
u_L_EF_L = (1/l_EF_L.length())*l_EF_L

u_L_FG_R = (1/l_FG_R.length())*l_FG_R
u_L_FG_L = (1/l_FG_L.length())*l_FG_L


pV1_0 = pAB - lA*u_L_BE_R
pV2_0 = pCD - lA*u_L_BE_L

pV3_0 = pAB - 5*lA*u_L_BE_R + l_3_length*u_L_BE_R
pV4_0 = pCD - 5*lA*u_L_BE_L + l_4_length*u_L_BE_L

pV5_0 = pER - 5*lA*u_L_EF_R
pV6_0 = pEL - 5*lA*u_L_EF_L

pV7_0 = pFR - 5*lA*u_L_FG_R
pV8_0 = pFL - 5*lA*u_L_FG_L

pV9_0  = pGR - 5*lA*G.y
pV10_0 = pGL - 5*lA*G.y



v_l1 = pV1_0.time_derivative().dot(u_L_BE_R)
v_l2 = pV2_0.time_derivative().dot(u_L_BE_R)
v_l3 = pV3_0.time_derivative().dot(u_L_BE_L)
v_l4 = pV4_0.time_derivative().dot(u_L_BE_L)

v_l5 = pV5_0.time_derivative().dot(u_L_EF_R)
v_l6 = pV6_0.time_derivative().dot(u_L_EF_L)
v_l7 = pV7_0.time_derivative().dot(u_L_FG_R)
v_l8 = pV8_0.time_derivative().dot(u_L_FG_L)
v_l9 = pV9_0.time_derivative().dot(G.y)
v_l10 = pV10_0.time_derivative().dot(G.y)

f1 = sympy.Symbol('f1')
f2 = sympy.Symbol('f2')
f3 = sympy.Symbol('f3')
f4 = sympy.Symbol('f4')

fR = sympy.Symbol('fR')
fL = sympy.Symbol('fL')

cond1 = {}
cond1[lA] = 0.05
cond1[lh] = 0.05
cond1[lT] = 1
cond1[Fx_tip] = 0
cond1[Fy_tip] = -10
cond1[T_tip] = 0

# T1_virtual_l1 = pGF - pNG
# T1_virtual_l2 = pBD - pGF

# T1_virtual_q1 = qG
# T1_virtual_q2 = sympy.tan(T1_virtual_l1.dot(T1_virtual_l2)/(T1_virtual_l1.length()*T1_virtual_l2.length()))

v
# def calculate_triangle_force(q1,q2,vR,vL,cond1):
q1,q2,vR,vL = qG,qF,v_l5,v_l6
q_T1 = sympy.Matrix([q1,q2])
J_T1 = v.jacobian(q_T1)

f = sympy.Matrix([Fx_tip,Fy_tip,T_tip])
T_ind = J_T1.T*f

v_t_t1 = sympy.Matrix([vR,vL])

J_t_ind_T1 = v_t_t1.jacobian(q_T1)

# f_tip = sympy.Matrix([Fx_tip,Fy_tip,T_tip])   

f_t_T1_sym = sympy.Matrix([fR,fL])
ft1_T1 = (J_t_ind_T1.T)*f_t_T1_sym
ft_error_T1 = T_ind-ft1_T1
ft_error_sym_T1 = ft_error_T1.subs(initialvalues).subs(cond1)
    # return ft_error_sym_T1

# ft_error_sym_T1 = calculate_triangle_force(qG,qE,v_l5,v_l6,cond1)

def calculate_f_dump(x1):
    value3 = numpy.sum(numpy.asanyarray(x1)**2)*1
    return value3

bounds1 = [(-1e3,1e3),(-1e3,1e3)]

A_eq  = numpy.array ( ft_error_sym_T1.jacobian(sympy.Matrix([fR,fL]))).astype(numpy.float64)
lb1 = -numpy.array(ft_error_sym_T1.subs({f1:0,f2:0,f3:0,f4:0,fR:0,fL:0})).astype(numpy.float64)
ub1 = -numpy.array(ft_error_sym_T1.subs({f1:0,f2:0,f3:0,f4:0,fR:0,fL:0})).astype(numpy.float64)
lb = numpy.transpose(lb1).reshape(2) - 1e-4
ub = numpy.transpose(ub1).reshape(2) + 1e-4
con1 = LinearConstraint(A_eq, lb, ub)

# res = dual_annealing(calculate_f_dump,bounds1)
res = minimize(calculate_f_dump,[1,-1],bounds=bounds1,constraints=con1,method='SLSQP',options={'disp':True})

print(res.x)

