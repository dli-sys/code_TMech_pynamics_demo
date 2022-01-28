
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
    ax2 = draw_skeleton(ini0, [pNA,pAB,pBD,pCD,pNA],linestyle='-',color='k',displacement=displacement,amplify=amplify)
    return ax2,initialvalues

# ax1 = plot_one_config(75,0,[0,0])
# ax1 = plot_one_config(75,0,[0.05,0])
# ax1 = plot_one_config(75,0,[0.05,0.05])
plt.close('all')

num = 4
angle1 = 30
angle2 = 75

angle_tilt_s = numpy.linspace(30, -30,5)
angles = numpy.linspace(angle1,angle2,num)

angle_tilt_s1 = numpy.linspace(30, -30,5)
angles1 = numpy.linspace(30, 75,4)
angle_2d1,angle_2d2=numpy.meshgrid(angles1,angle_tilt_s1)

t_maxs_2d = []

fig, (ax1, ax2) = plt.subplots(2)
# ax2 = plt.subplot(222)
# ax3 = plt.subplot(223)
import random
# for item in angle_tilt_s:
#     t_maxs = []
#     for item1 in angles:
for item in range(0,5):
    t_maxs = []
    for item1 in range(0,4):
        # t_max = (item1**2-item**3)/10000 +10*random.random()
        #Draw config
        x_dis = angle_tilt_s1[item]
        y_dis = angles1[item1]
        
        q1_value = angles[item1]
        ori_value = angle_tilt_s[item]
        # print("%.0f" %(q1_value))
        # print("%.0f" %(ori_value))
        ax1,initialvalues1 = plot_one_config(q1_value,ori_value,[x_dis,y_dis],amplify=100)
        #calcualte max torque
        T_ind_sym = ft1.subs(initialvalues1).subs(cond1).T
        T_s = T_ind_sym.subs({f1:0,f4:0})
        T1 = T_s[0]+T_s[1]
        max_fric  = 1.56
        # bounds1 = [(0,max_fric),(0,max_fric)]
        bounds1 = [(-max_fric,0),(-max_fric,0)]
        # t_max = minimize(lambda x:T1.subs({f2:x[0],f3:x[1]}),[0,0],bounds=bounds1)
        t_max = T1.subs({f2:0,f3:-1.56})
        # print(f'tendon force: {t_max.x}')
        # t_maxs = numpy.append(t_maxs,t_max.fun)  
        t_maxs = numpy.append(t_maxs,t_max)  
        #add text
        error_string = "%.4f" % (t_max) +", " +'%.4f' % (t_max)+ ",\n "+"0.01"   
        error_string = "%.4f" % (t_max/0.03)
        # plt.text(item,item1,error_string)
        plt.text(x_dis,y_dis-4,error_string,ha='center',va='top')        
        plt.plot(x_dis,y_dis,'o', color='k',markersize=abs(t_max*500),alpha=0.5)
    if ori_value == angle_tilt_s[0]:
        t_maxs_2d = t_maxs
    else:
        t_maxs_2d = numpy.vstack((t_maxs_2d,t_maxs))
  
        
#generate cmap
import matplotlib.colors
ax2=plt.subplot(111)
# fig=plt.figure()
# plt.ioff()
# ax1,initialvalues1 = plot_one_config(item,item1,[item,item1],amplify=100)
# ax2 = fig.add_subplot()
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("cdict1", ["blue1","pink1","red1"])
# ax2 = fig.add_subplot()     
   
im = ax2.pcolormesh(angle_2d2,angle_2d1,t_maxs_2d.astype('float'),shading='gouraud',cmap='coolwarm')
# fig.colorbar(im)
# ax1.gca().invert_yaxis()
# plt.gca().invert_xaxis()
plt.rcParams["font.family"] = "Times New Roman"
ax1.axis('equal')
ax1.set_ylabel("Inner angle $q_{AC}$($^{\circ}$)",fontsize=10)
ax1.set_xlabel("Orientation $q_a$($^{\circ}$)",fontsize=10)
plt.title("Max torque $T_{tip}$($Nm$)",fontsize=10)

# ax1.set_xlim([-35,30])
# ax1.set_ylim([20,130])
# ax1.set_xbound([-35,30])
# ax1.set_ybound([20,130])
# ax1.set_aspect(1)
ax2.set_xlim([40,40])
ax2.set_ylim([130,2])


# ax1.set_xticks([30,15,0,15,-30])
# ax1.set_yticks([120,90,60,30])
ax2.grid('on')
plt.show()

