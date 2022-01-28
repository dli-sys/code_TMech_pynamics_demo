#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:13:17 2021

Written by Daniel M. Aukes and Dongting Li
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

from sympy import sin,cos,tan,sqrt,acos

from scipy.optimize import minimize
from scipy.optimize import shgo
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

import math

import sympy
import numpy
import matplotlib.pyplot as plt
# plt.close('all')
plt.ion()

from math import pi

def draw_skeleton(ini0,points1,fig,linestyle='solid'):
    # points1 = [pGR,pFR,pER,pAB]
    po2 = PointsOutput(points1, constant_values=system.constant_values)
    po2.calc(numpy.array([ini0,ini0]),[0,1])
    po2.plot_time(fig=fig,linestyle=linestyle)


def gen_system(para):
    
    angle_d1 = para[0]
    angle_t1 = para[1]
    angle_t2 = para[2]
    angle_t3 = para[3]
    angle_t4 = para[4]
    l_d1 = para[5]
    l_t_w = para[6]
    l_t_h = para[7]
    nth = para[8]
    
    # angle_d1 = 2*pi/3
    # angle_t1 = 0
    # angle_t2 = pi/8
    # angle_t3 = pi/3
    # angle_t4 = -pi/6
    # l_d1 = 0.01
    # l_t_w = 0.06
    # l_t_h = 0.1
    # nth=2

    lA = Constant(l_d1,'lA',system)
    lh = Constant(l_t_h,'lh',system)
    lT = Constant(l_t_w,'lT',system)
    
    m = Constant(1,'m',system)
    
    g = Constant(9.81,'g',system)
    b = Constant(1e0,'b',system)
    k = Constant(1e1,'k',system)
    tol = 1e-5
    tinitial = 0
    tfinal = 5
    tstep = 1/30
    t = numpy.r_[tinitial:tfinal:tstep]
    
    
    Fx_tip = sympy.Symbol('Fx_tip')
    Fy_tip= sympy.Symbol('Fy_tip')
    T_tip= sympy.Symbol('T_tip')
    
    qF,qF_d,qF_dd = Differentiable('qF',system)
    # qG,qG_d,qG_dd = Differentiable('qG',system)
    
    
    # [angle_d1,angle_t4,angle_t1,angle_t2,angle_t3,l_d1,l_t_w,l_t_h,nth] = inputStates
    
    l_d_h = 2*l_d1*cos(angle_d1/2)
    
    T1_x = 0
    T1_y = l_t_h
    
    T2_x =  -l_t_h*sin(angle_t2)
    T2_y =  l_t_h*(cos(angle_t2) + 1)
    # T3_x = T2_x+l_t_h*cos(angle_t3)*sin(angle_t2) - l_t_h*sin(angle_t3)*cos(angle_t2)
    # T3_y = T2_y+l_t_h*cos(angle_t3)*cos(angle_t2) + l_t_h*sin(angle_t3)*sin(angle_t2)
    # D1_x = T3_x+l_d_h*sin(angle_t2+angle_t3+angle_t4)
    # D1_y = T3_y+l_d_h*cos(angle_t2+angle_t3+angle_t4)
    T3_x = -l_t_h*(sin(angle_t2 + angle_t3) + sin(angle_t2))
    T3_y =  l_t_h*(cos(angle_t2 + angle_t3) + cos(angle_t2) + 1)
    D1_x =  - l_t_h*sin(angle_t2 + angle_t3) - l_t_h*sin(angle_t2) - l_d_h*sin(angle_t2 + angle_t3 + angle_t4)
    D1_y =   l_t_h + l_t_h*cos(angle_t2 + angle_t3) + l_t_h*cos(angle_t2) + l_d_h*cos(angle_t2 + angle_t3 + angle_t4)
    
    L_tip = sqrt(D1_x**2+D1_y**2)
    
    initialvalues = {}
    
    # initialvalues[qG_d] = 0
    initialvalues[qF_d] = 0
    
    if nth==1:
        L_a = l_t_h
        L_b = sqrt((D1_x-T1_x)**2+ (D1_y-T1_y)**2)
        # initialvalues[qG] = angle_t1
        initialvalues[qF] = angle_t1
    elif nth==2:
        L_a = l_t_h
        L_b = sqrt((D1_x-T1_x)**2+ (D1_y-T1_y)**2)
        # initialvalues[qG] = (angle_t1)
        initialvalues[qF] = math.atan2((D1_x-T1_x),(D1_y-T1_y))
    elif nth==3:
        L_a = sqrt(T2_x**2+ T2_y**2)
        L_b = sqrt((D1_x-T2_x)**2+ (D1_y-T2_y)**2)
        # initialvalues[qG] = (0.5*(angle_t2))
        initialvalues[qF] = math.atan2((D1_x-T2_x),(D1_y-T2_y))
    
    statevariables = system.get_state_variables()
    
    N = Frame('N',system)
    F = Frame('F',system)
    # G = Frame('G',system)
    
    # V1 = Frame('V_1',system)
    # V2 = Frame('V_2',system)
    
    system.set_newtonian(N)
    
    F.rotate_fixed_axis(N,[0,0,1],qF,system)
    # G.rotate_fixed_axis(F,[0,0,1],qF,system)
    # F.rotate_fixed_axis(G,[0,0,1],qF,system)
    
    
    pNG = 0*N.y
    
    if nth==2: 
        pGF = pNG + L_a*N.y
        pFE = pGF + L_b*F.y
        
        pFM = pNG +L_a*N.y
    
        pEM = pFM + lh*(sin(angle_t2))*N.x + lh*(cos(angle_t2))*N.y
        pEM2 = pEM +lh*(sin(angle_t2+angle_t3))*N.x + lh*(cos(angle_t2+angle_t3))*N.y    
        
        pDL = pEM2 + l_d_h*sin(-angle_d1/2+angle_t2+angle_t3+angle_t4)/2*N.x +l_d_h*cos(-angle_d1/2+angle_t2+angle_t3+angle_t4)/2*N.y
        pDR = pEM2 + l_d_h*sin(angle_d1/2+angle_t2+angle_t3+angle_t4)/2*N.x +l_d_h*cos(angle_d1/2+angle_t2+angle_t3+angle_t4)/2*N.y        
        
        pXR =  pEM + 0.5*cos(angle_t2+angle_t3)*lT*N.x - 0.5*sin(angle_t2+angle_t3)*lT*N.y
        pXL =  pEM - 0.5*cos(angle_t2+angle_t3)*lT*N.x + 0.5*sin(angle_t2+angle_t3)*lT*N.y
        
        # pDM = pEM2 - 0.5*sin(angle_t2+angle_t3+angle_t4)*l_d_h*N.x + 0.5*cos(angle_t2+angle_t3+angle_t4)*l_d_h*N.y
        # pDR = pDM + l_d_h*(sin(angle_t2-angle_d1/2+angle_t3+angle_t4)-sin(angle_d1/2+angle_t2+angle_t3+angle_t4))/4*N.x
        # pDR = pDR - l_d_h*(cos(angle_t2-angle_d1/2+angle_t3+angle_t4)-cos(angle_d1/2+angle_t2+angle_t3+angle_t4))/4*N.y
        # pDL = pDM - l_d_h*(sin(angle_t2-angle_d1/2+angle_t3+angle_t4)-sin(angle_d1/2+angle_t2+angle_t3+angle_t4))/4*N.x
        # pDL = pDL - l_d_h*(cos(angle_t2-angle_d1/2+angle_t3+angle_t4)-cos(angle_d1/2+angle_t2+angle_t3+angle_t4))/4*N.y
        # pEM = pFM + lh*(sin(angle_t2))*N.x + lh*(cos(angle_t2))*N.y
        # pEM2 = pEM +lh*(sin(angle_t2+angle_t3))*N.x + lh*(cos(angle_t2+angle_t3))*N.y        
        pFR1 = pFM + 0.5*cos(angle_t2)*lT*N.x - 0.5*sin(angle_t2)*lT*N.y
        pFL1 = pFM - 0.5*cos(angle_t2)*lT*N.x + 0.5*sin(angle_t2)*lT*N.y
        
        L_s1 = 2*l_t_h*cos(angle_t3/2)
        if round(L_s1**2 + L_b**2-l_d_h**2,6) == round(2*L_b*L_s1,6):
            angle_e1=0
        else:
            angle_e1 = acos((L_s1**2 + L_b**2-l_d_h**2)/(2*L_b*L_s1))
        if angle_t3 <0:
            angle_e2 = -angle_e1 + angle_t3/2
        else:
            angle_e2 = angle_e1 + angle_t3/2
               
        pFR = pFM  + 0.5*lT*cos(angle_e2)*F.x + 0.5*lT*sin(angle_e2)*F.y
        pFL = pFM  - 0.5*lT*cos(angle_e2)*F.x - 0.5*lT*sin(angle_e2)*F.y
    
        pGM= pNG
        pGR = pGM + 0.5*lT*N.x
        pGL = pGM - 0.5*lT*N.x
    
    elif nth==3:
        
        pGF = pNG + L_a*cos(0.5*(angle_t2))*N.y + L_a*sin(0.5*(angle_t2))*N.x
        pFE = pGF + L_b*F.y
                        
        pFM = pGF
        pEM = pGF + lh*(sin(angle_t3+angle_t2))*N.x + lh*(cos(angle_t3+angle_t2))*N.y

        pDL = pEM + l_d_h*sin(-angle_d1/2+angle_t2+angle_t3+angle_t4)/2*N.x +l_d_h*cos(-angle_d1/2+angle_t2+angle_t3+angle_t4)/2*N.y
        pDR = pEM + l_d_h*sin(angle_d1/2+angle_t2+angle_t3+angle_t4)/2*N.x +l_d_h*cos(angle_d1/2+angle_t2+angle_t3+angle_t4)/2*N.y        


        pEM2=pFE
        pFR1 = pFM + 0.5*lT*cos(angle_t3+angle_t2)*N.x - 0.5*lT*sin(angle_t3+angle_t2)*N.y
        pFL1 = pFM - 0.5*lT*cos(angle_t3+angle_t2)*N.x + 0.5*lT*sin(angle_t3+angle_t2)*N.y
        
        if round(l_t_h**2 + L_b**2-l_d_h**2,6) == round(2*L_b*l_t_h,6):
            angle_e0=0
        else:
            angle_e0 = acos((l_t_h**2 + L_b**2-l_d_h**2)/(2*L_b*l_t_h))
        
        if angle_t4<0:
            angle_e1 = -angle_e0
        else:
            angle_e1 = angle_e0
        pFM = pGF
        pFR = pFM + 0.5*lT*cos(angle_e1)*F.x + 0.5*lT*sin(angle_e1)*F.y
        pFL = pFM - 0.5*lT*cos(angle_e1)*F.x - 0.5*lT*sin(angle_e1)*F.y
        
        
        # pEM = pGF + lh*(sin(angle_t2/2 + angle_t3) )*N.x +lh*(cos( angle_t2/2+ angle_t3))*N.y
        # pEM2 = pFE
        # pFR = pGF + 0.5*lT*cos(angle_t4)*F.x - 0.5*lT*sin(angle_t4)*F.y
        # pFL = pGF - 0.5*lT*cos(angle_t4)*F.x + 0.5*lT*sin(angle_t4)*F.y        
        angle_e2 = angle_t2/2
        pGM = pNG + lh*(cos(angle_t1))*N.y + lh*(sin(angle_t1))*N.x
        pGR = pGM + 0.5*lT*cos(angle_t2)*N.x - 0.5*lT*sin(angle_t2)*N.y
        pGL = pGM - 0.5*lT*cos(angle_t2)*N.x + 0.5*lT*sin(angle_t2)*N.y
        
        pXR = pNG + 0.5*lT*N.x
        pXL = pNG - 0.5*lT*N.x
    

    
    points = [pNG,pGF,pFE]
    statevariables = system.get_state_variables()
    ini0 = [initialvalues[item] for item in statevariables]
           
    # po1 = PointsOutput(points, constant_values=system.constant_values)
    # po1.calc(numpy.array([ini0,ini0]),[0,1])
    # po1.plot_time()
    
    fig1 = plt.figure()
    
    draw_skeleton(ini0,[pNG,pGM],fig1)
    draw_skeleton(ini0,[pGM,pGF],fig1)
    draw_skeleton(ini0,[pFM,pEM,pEM2,pFE],fig1)   
    if nth==2:
        draw_skeleton(ini0,[pFM,pFE],fig1,linestyle='dashed')
        draw_skeleton(ini0,[pXR,pEM,pXL],fig1)
        draw_skeleton(ini0,[pEM2,pDR,pFE],fig1)
        draw_skeleton(ini0,[pEM2,pDL,pFE],fig1,linestyle='dashed')
    if nth==3:
        draw_skeleton(ini0,[pNG,pGF],fig1,linestyle='dashed')
        draw_skeleton(ini0,[pGF,pFE],fig1,linestyle='dashed')
        draw_skeleton(ini0,[pXR,pNG,pXL],fig1)
        draw_skeleton(ini0,[pEM,pDR,pFE],fig1)
        draw_skeleton(ini0,[pEM,pDL,pFE],fig1,linestyle='dashed')
    
    # draw_skeleton([pGL,pGM,pGR])
    # draw_skeleton([pFL,pFM,pFR])
    
    draw_skeleton(ini0,[pGL,pGM],fig1,linestyle='dashed')
    draw_skeleton(ini0,[pFL,pFM],fig1,linestyle='dashed')
    draw_skeleton(ini0,[pGM,pGR],fig1)
    draw_skeleton(ini0,[pFM,pFR],fig1)
    
    
    
    draw_skeleton(ini0,[pFL1,pFM],fig1,linestyle='dashdot')
    draw_skeleton(ini0,[pFM,pFR1],fig1,linestyle='dashdot')
    
    plt.grid()
    
    # pEcm = pFE + lA/2*E.y
    pFcm = pGF + L_b/2*F.y
    # pGcm = pNG + L_a/2*G.y
    
    
    # BodyE = Particle(pEcm,m,'ParticleE',system)
    BodyF = Particle(pFcm,m,'ParticleF',system)
    # BodyG = Particle(pGcm,m,'ParticleG',system)
    
    
    f,ma = system.getdynamics()
    dyn = sympy.Matrix(f)-sympy.Matrix(ma)
    # eq_dd = sympy.Matrix(eq_dd)
    
    vFE = pFE.time_derivative()
    # vNE = (pFE-pNG).time_derivative()
    wNB =  qF_d*N.z
    
    vx = vFE.dot(N.x)
    vy = vFE.dot(N.y)

    # sympy.Matrix([vx,vy])
    
    # wNB_scalar = sympy.Matrix([pFE.dot(N.x),pFE.dot(N.y)]).dot(sympy.Matrix([vx,vy]))/(L_tip**2)
    wNB_scalar = wNB.dot(N.z)
    v = sympy.Matrix([vx,vy,wNB_scalar])
    
    l_FG_R = pFR - pGR
    l_FG_L = pFL - pGL
    
    u_L_FG_R = (1/l_FG_R.length())*l_FG_R
    u_L_FG_L = (1/l_FG_L.length())*l_FG_L
    
    pV5_0 = pFR - 0.05*lA*u_L_FG_R
    pV6_0 = pFL - 0.05*lA*u_L_FG_L
    
    v_R = pV5_0.time_derivative().dot(u_L_FG_R)
    v_L = pV6_0.time_derivative().dot(u_L_FG_L)
    
    fR = sympy.Symbol('fR')
    fL = sympy.Symbol('fL')
    
    q_T1 = sympy.Matrix([qF_d])
    J_T1 = v.jacobian(q_T1)
    
    f = sympy.Matrix([Fx_tip,Fy_tip,T_tip])
    T_ind = J_T1.T*f
    
    v_t_t1 = sympy.Matrix([v_R,v_L])
    J_t_ind_T1 = v_t_t1.jacobian(q_T1)
    
    f_t_T1_sym = sympy.Matrix([fR,fL])
    ft1_T1 = (J_t_ind_T1.T)*f_t_T1_sym
    ft_error_T1 = (T_ind-ft1_T1).subs(initialvalues).subs(system.constant_values)
    T_ind_sym = T_ind.subs(initialvalues).subs(system.constant_values)
    max_T = ft1_T1.subs(initialvalues).subs(system.constant_values)
    
    # A_eq1 = numpy.array(max_T.jacobian(sympy.Matrix([fR,fL]))).astype(numpy.float64)
    max_fric = 5
    bounds1 = [(0,max_fric),(0,max_fric)]
    
    from scipy.optimize import minimize_scalar
    
    obj1=lambda f_input:-numpy.array(max_T.subs({fR:f_input[0],fL:f_input[1]})).astype(numpy.float64)[0][0]
    
    res = minimize(obj1,[0,0],bounds=bounds1,options={'disp':False})
    max_T_value = -obj1(res.x)
    # res = minimize(obj1,[0,0],bounds=bounds1,options={'disp':False})
    # max_T_value = obj1(res.x)
    
    return ft_error_T1,fR,fL,T_ind_sym,max_T_value
# cond1 = {}
# cond1[lA] = l_d1
# cond1[lh] = l_t_h
# cond1[lT] = l_t_w
# cond1[Fx_tip] = -10
# cond1[Fy_tip] = 0
# cond1[T_tip] = 0

def obj1(f_input):
    y=numpy.array(max_T.subs({fR:f_input[0],fL:f_input[1]})).astype(numpy.float64)[0][0]
    # y1 = y[0][0]
    return y

def calculate_f_dump(x1):
    value3 = numpy.sum(numpy.asanyarray(x1)**2)*1
    return value3

# bounds1 = [(-1e3,1e3),(-1e3,1e3)]
def calculate_force_angle(load):

    cond1 = {}
    # cond1[lA] = l_d1
    # cond1[lh] = l_t_h
    # cond1[lT] = l_t_w
    Fx_tip = sympy.Symbol('Fx_tip')
    Fy_tip= sympy.Symbol('Fy_tip')
    T_tip= sympy.Symbol('T_tip')
    cond1[Fx_tip] = load[0]
    cond1[Fy_tip] = load[1]
    cond1[T_tip] = load[2]
    tol=1e-4
    ft_error_sym_T1 = ft_error_T1.subs(cond1)
    T_ind_value = T_ind_sym.subs(cond1)
    # print(ft_error_sym_T1)
    # bounds1 = [(-1e3,1e3),(-1e3,1e3)]    
    bounds1 = [(0,1e3),(0,1e3)]
    # bounds1 = [(-1e3,0)]
    
    A_eq  = numpy.array(ft_error_sym_T1.jacobian(sympy.Matrix([fR,fL]))).astype(numpy.float64)
    lb1 = -numpy.array(ft_error_sym_T1.subs({fR:0,fL:0})).astype(numpy.float64)
    ub1 = -numpy.array(ft_error_sym_T1.subs({fR:0,fL:0})).astype(numpy.float64)
    lb = numpy.transpose(lb1).reshape(1) - tol
    ub = numpy.transpose(ub1).reshape(1) + tol
    con1 = LinearConstraint(A_eq, lb, ub)
    
    res = minimize(calculate_f_dump,[0,0],bounds=bounds1,constraints=con1,options={'disp':True})
    
    # max_T_value = max_T.subs({fR:1,fL:1})
    
    # print("T_ind value1")
    # print(T_ind.subs(initialvalues).subs(cond1))
    # print("T_ind value2")
    # print(((J_t_ind_T1.T).subs(initialvalues).subs(cond1)).dot(res.x))
    return [res.x,ft_error_sym_T1,T_ind_value]

# para = [1*pi/3,0,pi/6,pi/6,pi/6,0.01,0.06,0.1,2]   

para = [1*pi/3,0,pi/6,-pi/3,pi/6,0.05,0.1,0.08,3]  
 
system = System()
pynamics.set_system(__name__,system)
ft_error_T1,fR,fL,T_ind_sym,max_T = gen_system(para)
aa = calculate_force_angle([1,0,0])

angle_range = pi/2-(math.atan2(para[6]/2,para[7])+pi/36)
math.degrees(angle_range)
# 
num = 2
for item in numpy.linspace(-angle_range,angle_range,num):
    print(item)
    # print(calculate_force_angle(item))
    para[3] = item
    system = System()
    pynamics.set_system(__name__,system)
    ft_error_T1,fR,fL,T_ind_sym,max_T = gen_system(para)
    aa = calculate_force_angle([0,0,1])
    if item == -angle_range:
        values = aa[0]
        T_values = aa[-1]
        max_T_values = max_T
    else:
        values = numpy.vstack([values,aa[0]])
        T_values = numpy.vstack([T_values,aa[-1]])
        max_T_values = numpy.vstack([max_T_values,max_T])



fig, ax1 = plt.subplots()
ln1=ax1.plot(numpy.rad2deg(numpy.linspace(-angle_range,angle_range,num)),values,'-',label=(r"$f_R$",r"$f_L$"))
ax1.set_xlabel('Angle ($^{\circ}$)')
ax1.set_ylabel('Tendon Force (N)')
ax2 = ax1.twinx()
ln2=ax2.plot(numpy.rad2deg(numpy.linspace(-angle_range,angle_range,num)),max_T_values,color='r',linestyle='dashed',label=(r"$T$"))
ax2.set_ylabel('Max Holding Torque (Nm)')
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

# plt.figure()
# plt.plot(numpy.rad2deg(numpy.linspace(-angle_range,angle_range,num)),T_values)
# ax1.legend([r"$f_R$",r"$f_L$"])
# fig.legend([r"$f_R$",r"$f_L$",r"$T$"],loc='upper left')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
