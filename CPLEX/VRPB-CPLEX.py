# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 08:46:20 2022

@author: Irandokht
"""

import numpy as np
np.random.seed(1000)
import pandas as pd
from openpyxl.workbook import Workbook
import random
random.seed(1000)
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import copy
import matplotlib.pyplot as plt
plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=15)  # fontsize of the figure title
from docplex.mp.model import Model
###############################################################################
''' initialization'''
###############################################################################
method = 'CPLEX'   # CPLEX or CFRS
Path ='F:\\Research\\StaticVRPB\\Computational Experiments\\4-Classes-of-the-VRPB\\Datasets\\'


Instances = ['Charlotte_80_Q30','Lansing_80_Q30','Charlotte_40_Q10','Lansing_40_Q10', 'Charlotte_40_Q30','Lansing_40_Q30']
Sheets = [0, 20, 40, 60, 80, 100]  # other sheets names in each instance: 'DistanceMatrix', 'ShortestPath'

# for routing phase
Alpha = [1, 0] #alpha = 0 if the VRPB with mixed solution and 1 if the VRPB with backhaul solution

#time_limit = 3600
###############################################################################
''' reading data '''
###############################################################################
def Data(instance, sheet):
    global V, V_0, L, L_0, B, B_0
    global A, A_L, A_B, A_C1, A_C2, A_C3
    global Q, k
    global d, p, c
    global ins, Graph
    
    
    xls = pd.ExcelFile(str(Path)+str(instance)+'.xlsx')
    ins = pd.read_excel(xls, str(sheet))
    distance = pd.read_excel(xls, sheet_name='DistanceMatrix', index_col=0) 
    
    Graph = ins['Graph'].iloc[0]
    
    # 0 represents depot
    V = list(ins[ins['type']!=0]['node_id'])
    V_0 = [0] + V
    
    # linehaul customers ID
    L = list(ins[ins['type']==1]['node_id'])
    L_0 = [0] + L
    
    # backhaul customers ID
    B = list(ins[ins['type']==2]['node_id'])
    B_0 = B + [0]
    
    # truck capacity and No. of avaiable trucks
    Q = ins['Q'].iloc[0]    
    k = int(ins['k'].iloc[0])
    
    # all links, completed graph
    A_L = [(i,j) for i in L_0 for j in L if i!=j]
    A_B = [(i,j) for i in B for j in B_0 if i!=j]
    A_C1 = [(i,j) for i in L for j in B_0]
    A_C2 = [(i,j) for i in B for j in L]
    A_C3 = [(0,j) for j in B]
    
    A = A_L + A_B + A_C1 + A_C2 + A_C3
    
    
    # distance is computed as the shorthest path for every two nodes based on Haversine distance metric in miles and rounded to 2 decimals     
    c = {(i,j):round(distance[ins.loc[ins['node_id']==i, 'node_id_prev'].iloc[0]][ins.loc[ins['node_id']==j, 'node_id_prev'].iloc[0]],2) for i in V_0 for j in V_0}
    
    
    # demand and pickup of all nodes, demand/pickup is zero for the depot
    d = {i: ins[ins['node_id']==i]['delivery'].iloc[0] for i in V_0}
    p = {i: ins[ins['node_id']==i]['pickup'].iloc[0] for i in V_0}
###############################################################################
'''get the sequence of vertex/node ID for optimized routes strating and ending from depot'''
###############################################################################
def findTuple(elem, arcs):
  for t in arcs:
    if t[0] == elem:
      return t
  return None
def node_seq(arcs, start_depot, end_depot):
    startRoutes = list(filter(lambda elem: elem[0]==start_depot, arcs))
    sequence = list()
    for i in range(len(startRoutes)):
      tempList = list()
      currentTuple = startRoutes[i]
      tempList.append(currentTuple[0])
      tempList.append(currentTuple[1])
      while True:
        if currentTuple[1] == end_depot:
          break
        else:
          nextTuple = findTuple(currentTuple[1], arcs)
          currentTuple = nextTuple
          tempList.append(currentTuple[1])
      sequence.append(tempList)
    return  sequence
###############################################################################
''' solve vehicle routing problem with backhauls for small scale problems using CPLEX '''
###############################################################################
def VRPB(alpha):
    mdl = Model('VRPB')
    
    # decision variables
    x = mdl.binary_var_dict(A_L, name = 'x')
    y = mdl.binary_var_dict(A_B, name = 'y')
    z1 = mdl.binary_var_dict(A_C1, name = 'z1')
    z2 = mdl.binary_var_dict(A_C2, name = 'z2')
    z3 = mdl.binary_var_dict(A_C3, name = 'z3')
          
    u = mdl.continuous_var_dict(A, ub = Q, name = 'u')
    w = mdl.continuous_var_dict(A, ub = Q, name = 'w')
    
    # degree constraints
    mdl.add_constraint(mdl.sum(x[0,j]for j in L) + (1-alpha)*mdl.sum(z3[0,j] for j in B)==k)
    mdl.add_constraint(mdl.sum(z1[i,0]for i in L) + mdl.sum(y[i,0]for i in B)==k)
    
    mdl.add_constraints(mdl.sum(x[i,j]for i in L_0 if i!=j) + (1-alpha)*mdl.sum(z2[i,j]for i in B)==1 for j in L)
    mdl.add_constraints(mdl.sum(x[i,j]for j in L if i!=j) + mdl.sum(z1[i,j]for j in B_0)==1 for i in L)
    
    mdl.add_constraints(mdl.sum(z1[i,j]for i in L) + mdl.sum(y[i,j]for i in B if i!=j) + (1-alpha)*z3[0,j]==1 for j in B)
    mdl.add_constraints((1-alpha)*mdl.sum(z2[i,j]for j in L)+ mdl.sum(y[i,j]for j in B_0 if i!=j)==1 for i in B)
    
    # capacity and connectivity constraints
    mdl.add_constraints(mdl.sum(u[j,i] for j in V_0 if i!=j) - mdl.sum(u[i,j] for j in V_0 if i!=j) == d[i] for i in V)
    mdl.add_constraints(mdl.sum(w[i,j] for j in V_0 if i!=j) - mdl.sum(w[j,i] for j in V_0 if i!=j) == p[i] for i in V)
    
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*x[i,j] for i,j in A_L)
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*y[i,j] for i,j in A_B)
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*z1[i,j] for i,j in A_C1)
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*(1-alpha)*z2[i,j] for i,j in A_C2)
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*(1-alpha)*z3[i,j] for i,j in A_C3)
    
    
    # objective function
    obj_1 = mdl.sum(c[i,j]*x[i,j]for i,j in A_L)
    obj_2 = mdl.sum(c[i,j]*y[i,j]for i,j in A_B)
    obj_3 = mdl.sum(c[i,j]*z1[i,j]for i,j in A_C1)
    obj_4 = (1-alpha)*mdl.sum(c[i,j]*z2[i,j]for i,j in A_C2)
    obj_5 = (1-alpha)*mdl.sum(c[i,j]*z3[i,j]for i,j in A_C3)
    obj = obj_1 + obj_2 + obj_3 + obj_4 + obj_5
    mdl.add_kpi(obj, 'VRPMB Cost')
    
    # solution
    mdl.minimize(obj)
    #mdl.parameters.timelimit = time_limit 
    solution = mdl.solve(log_output = False) #true if you need to see the steps of slover
    #mdl.report_kpis()
    mdl.export_as_lp()
    #print(mdl.export_as_lp())
    #print(solution)
    if not solution:
        print('fail in solving VRPB, there is no feasible solution')
    x_opt = [a for a in A_L if x[a].solution_value> 0.9]
    y_opt = [a for a in A_B if y[a].solution_value> 0.9]
    z1_opt = [a for a in A_C1 if z1[a].solution_value> 0.9]
    z2_opt = [a for a in A_C2 if z2[a].solution_value> 0.9]
    z3_opt = [a for a in A_C3 if z3[a].solution_value> 0.9]
    obj_opt = round(solution.objective_value,2)
    
    arcs = x_opt + y_opt + z1_opt + z2_opt + z3_opt
    route = node_seq(arcs, 0, 0)
    return obj_opt, route
###############################################################################
''' main algorithm'''
###############################################################################
headers = ['Graph', 'C', 'V', '% Simultaneous demand', 'LB', 'L', 'B', 'Q', 'k',
           'Solution', 'Solution method', 'itinerary', 'time (sec)','cost', 'date']

workbook_name = str(method)+'_'+str(datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.xlsx'
wb = Workbook()
page = wb.active
page.title = str(method)
page.append(headers)

for data in Instances:
    for sheet in Sheets:
        
        Data(str(data), str(sheet))
        
        start = time.time()       
        for alpha in Alpha:
            if alpha == 0:
                Solution = 'Mixed'
            elif alpha == 1:
                Solution = 'Backhaul'
            print('*************\n', data, sheet, Solution )
            
            cost, routes = VRPB(alpha)
            
            itinerary = {}
            for i in range(k):                
                itinerary[i+1] = routes[i]
            
            # compute total transportation cost
            total_cost = 0
            for keys in itinerary:
                rrr = itinerary[keys]
                for i in range(len(rrr)-1):
                    total_cost += c[rrr[i], rrr[i+1]]
    
            '''compute CPU running time in second'''
            stop = time.time()
            cpu_time = round(stop - start, 2)
            
            VV = list(set(ins[ins['type']!=0]['node_id_prev']))
            info = [str(Graph), len(VV), len(V), sheet, len(V)-len(VV), len(L), len(B), Q, k,
                    str(Solution), str(method), str(itinerary), cpu_time, round(total_cost,2),
                    str(datetime.now().strftime('%Y-%m-%d'))]
                        
                    
            page.append(info)
            wb.save(filename = workbook_name)












