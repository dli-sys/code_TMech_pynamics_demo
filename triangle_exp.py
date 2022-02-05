#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:13:17 2021

Written by Daniel M. Aukes and Dongting Li
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

def draw_skeleton(ini0,points1,linestyle='solid',color=[],displacement=[0,0],amplify=1):
    po2 = PointsOutput(points1, constant_values=system.constant_values)
    po2.calc(numpy.array([ini0,ini0]),[0,1])
    ax = po2.plot_time_c(newPlot=False,linestyle=linestyle,color=color,displacement=displacement,amplify=amplify)
    return ax
def plot_one_config(angle_value,displacement=[0,0],amplify=1,side='r'):
    initialvalues = {}   
    initialvalues[qF]   =(angle_value)*pi/180
    initialvalues[qF_d] =0*pi/180   
    statevariables = system.get_state_variables()
    ini0 = [initialvalues[item] for item in statevariables]  
    ax2 = draw_skeleton(ini0,[pGF,pNR,pNL,pGF,pGL,pFE,pGR,pGF],linestyle='-',color='k',displacement=displacement,amplify=amplify)
    
    if side=='r':
        draw_skeleton(ini0,[pNR,pGR],linestyle='--',color='k',displacement=displacement,amplify=amplify)
        draw_skeleton(ini0,[pNL,pGL],linestyle='-',color='b',displacement=displacement,amplify=amplify)
        draw_skeleton(ini0,[pFE,p_tip_L],linestyle='-',color=[0.5,0.5,0.5],displacement=displacement,amplify=amplify)
    if side=='l':
        draw_skeleton(ini0,[pNR,pGR],linestyle='-',color='r',displacement=displacement,amplify=amplify)
        draw_skeleton(ini0,[pNL,pGL],linestyle='--',color='k',displacement=displacement,amplify=amplify)
        draw_skeleton(ini0,[pFE,p_tip_R],linestyle='-',color=[0.5,0.5,0.5],displacement=displacement,amplify=amplify)

    # draw_skeleton(ini0, [pGR,pV5_0],linestyle='dashed')
    # draw_skeleton(ini0, [pGL,pV6_0],linestyle='dashed')    
    return initialvalues


system = System()
pynamics.set_system(__name__,system)

    
l_t_w = 0.085
l_t_h = 0.0736


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

    
initialvalues = {}

initialvalues[qF_d] = 0

L_a = l_t_h
L_b = l_t_h
initialvalues[qF] = 0

statevariables = system.get_state_variables()

N = Frame('N',system)
F = Frame('F',system)

system.set_newtonian(N)

F.rotate_fixed_axis(N,[0,0,1],qF,system)

pNG = 0*N.y

pGF = pNG + L_a*N.y
pFE = pGF + L_b*F.y    
       
pGR = pGF+ 0.5*lT*F.x
pGL = pGF- 0.5*lT*F.x

pNR = pNG + 0.5*lT*N.x
pNL = pNG - 0.5*lT*N.x


statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]
      
    
plt.grid()

pFcm = pGF + L_b/2*F.y

BodyF = Particle(pFcm,m,'ParticleF',system)


f,ma = system.getdynamics()
dyn = sympy.Matrix(f)-sympy.Matrix(ma)


vFE = pFE.time_derivative()
# vNE = (pFE-pNG).time_derivative()
wNB =  qF_d*N.z

vx = vFE.dot(N.x)
vy = vFE.dot(N.y)

wNB_scalar = wNB.dot(N.z)
v = sympy.Matrix([vx,vy,wNB_scalar])

l_FG_R = pGR - pNR
l_FG_L = pGL - pNL

u_L_FG_R = (1/l_FG_R.length())*l_FG_R
u_L_FG_L = (1/l_FG_L.length())*l_FG_L

pV5_0 = pGR - 2*lh*u_L_FG_R
pV6_0 = pGL - 2*lh*u_L_FG_L

v_R = pV5_0.time_derivative().dot(u_L_FG_R)
v_L = pV6_0.time_derivative().dot(u_L_FG_L)

l_tip_L = pGL - pFE
l_tip_R = pGR - pFE

u_l_tip_L  = (1/l_tip_L.length())*l_tip_L
u_l_tip_R  = (1/l_tip_R.length())*l_tip_R

p_tip_L = pFE - 0.75*lT*u_l_tip_L
p_tip_R = pFE - 0.75*lT*u_l_tip_R
p_tip_L_m = pFE - 0.75/2*lT*u_l_tip_L
p_tip_R_m = pFE - 0.75/2*lT*u_l_tip_R


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
max_T_halftway = ft1_T1.subs(system.constant_values)
plt.close('all')
max_fric =1.7
bounds1 = [(0,max_fric),(0,max_fric)]

angle_start = -45
angle_end = 45
num = 30

t_max1 = []
t_max2 = []

from scipy import interpolate

data_dir = os.path.join(os.getcwd(),'exp_data')
t_forces1 = numpy.genfromtxt(os.path.join(data_dir,'triangle_force_exp_forplot.csv'),delimiter=',')
t_forces = t_forces1*-0.035
# t_force_temp = t_forces[:,0]
t_force_temp_max = numpy.amax(t_forces,axis=0)
t_force_temp_min = numpy.amin(t_forces,axis=0)
t_force_temp_avg = numpy.average(t_forces,axis=0)
exp_angles = numpy.linspace(angle_start,angle_end,7)
ft_max = interpolate.interp1d(exp_angles,t_force_temp_max,fill_value = 'extrapolate', kind='quadratic')
ft_min = interpolate.interp1d(exp_angles,t_force_temp_min,fill_value = 'extrapolate', kind='quadratic')
ft_avg = interpolate.interp1d(exp_angles,t_force_temp_avg,fill_value = 'extrapolate', kind='quadratic')


for item in numpy.linspace(angle_start,angle_end,num):
    initialvalues = {}   
    initialvalues[qF]   =(item)*pi/180
    initialvalues[qF_d] =0*pi/180
    
    max_T = max_T_halftway.subs(initialvalues)
    obj1=lambda f_input:(max_T.subs({fR:f_input[0],fL:f_input[1]}))[0]
    obj2=lambda f_input:(-max_T.subs({fR:f_input[0],fL:f_input[1]}))[0]
    res1 = minimize(obj1,[0,0],bounds=bounds1,options={'disp':False})
    res2 = minimize(obj2,[0,0],bounds=bounds1,options={'disp':False}) 
    max_T_value1 = res1.fun
    max_T_value2 = res2.fun
    t_max1 = numpy.append(t_max1,max_T_value1)
    t_max2 = numpy.append(t_max2,max_T_value2)

# fig, ax1 = plt.subplots(111)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "15"

plt.plot(numpy.linspace(angle_start,angle_end,num),t_max1*1000,'b--',label='Left side Sim')
plt.plot(numpy.linspace(angle_start,angle_end,num),t_max2*1000,'r--',label='Right side Sim')

sim_angles = numpy.linspace(angle_start,angle_end,num)
plt.fill_between(exp_angles,ft_max(exp_angles)*1000,ft_min(exp_angles)*1000,color='b',alpha=0.25)
plt.plot(exp_angles,t_force_temp_avg*1000,'b',marker='d',label='Experiments')
plt.plot(exp_angles,ft_max(exp_angles)*1000,color='b',alpha=0.4,label='_nolegend_')
plt.plot(exp_angles,ft_min(exp_angles)*1000,color='b',alpha=0.4,label='_nolegend_')

T_loc = (t_max1+t_max2)/2
ft0  = interpolate.interp1d(numpy.linspace(angle_start,angle_end,num),T_loc,fill_value = 'extrapolate', kind='quadratic')
ft1 = interpolate.interp1d(numpy.linspace(angle_start,angle_end,num),t_max1,fill_value = 'extrapolate', kind='quadratic')

for item in numpy.linspace(angle_start,angle_end,7):
    initialvalues = {}   
    initialvalues[qF]   =(item)*pi/180
    initialvalues[qF_d] =0*pi/180
    dis_x = item
    dis_y = ft0(angle_start+angle_end)/2
    plot_one_config(item,displacement=[dis_x,dis_y*1000-70],amplify=100,side='l') 
    
    # error_string1 = "%.7f" % (ft1(item)/0.035)
    # plt.text(dis_x,dis_y*1000,error_string1,ha='center',va='top') 
    
    # plot_one_config(item,displacement=[dis_x,dis_y*1000-0],amplify=100,ax=[]) 
    # t_max = numpy.append(t_max,max_T_value)

for item in numpy.linspace(angle_start,angle_end,7):
    initialvalues = {}   
    initialvalues[qF]   =(item)*pi/180
    initialvalues[qF_d] =0*pi/180
    dis_x = item
    dis_y =  ft0(angle_start+angle_end)/2
    plot_one_config(item,displacement=[dis_x,dis_y*1000+8],amplify=100,side='r') 


plt.xlim([-60,60])
plt.show()
plt.xticks(exp_angles)

ax1 = plt.gca()
ax1.grid()

ax1.set_ylim(numpy.asarray(ax1.get_ylim())+[0,13])

ax1.set_ylabel("Max Torque (Nm)")
ax1.set_xlabel("Joint angle ($^{\circ}$)")

# plt.legend({"Left Side Sim","Right Side Sim","Error","Left Side Exp Average"},loc='upper right')
plt.legend(loc='upper center',ncol=3, )
ax1.set_yticks(numpy.linspace(-20,-75,11))
ax1.set_xticks(numpy.linspace(-45,45,7))

ax1.set_yticklabels( [(item/1000) for item in numpy.linspace(-20,-75,11)])
ax1.set_xticklabels([(item) for item in numpy.linspace(45,-45,7)])
