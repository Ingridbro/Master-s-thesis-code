from pyomo.environ import *
import numpy as np

"""This code is based on the MATLAB code Windhydtools by Magnus Korpås, 2004
    that can be found here https://www.ntnu.no/ansatte/magnus.korpas"""

def readdata():
    #Spotprice 2022 and 2021 from NordPool
    spot_price_data_2022 = np.loadtxt("../../../Hydrogen/Nord_Pool_Spot_Price_Trondheim_2022.txt")
    spot_price_data_2021 = np.loadtxt("../../../Hydrogen/Nord_Pool_Spot_Price_Trondheim_2021.txt")

    #Marginal loss rate - 6% during winter (01.11-01.05) and 4% during summer from TENSIO
    ma_loss_rt = np.loadtxt("../../../Hydrogen/ma_loss_rt.txt")

    #Electricity demand 2022 and 2021
    El_demand_2022 = np.loadtxt("../../../Hydrogen/el_demand_2022.txt")
    El_demand_2021 = np.loadtxt("../../../Hydrogen/el_demand_2021.txt")

    #Simulated PV production from PVSyst (TMY)
    E_PV_STO = np.loadtxt("../../../Hydrogen/PV_STO.txt")
    E_PV_TIP = np.loadtxt("../../../Hydrogen/PV_TIP.txt")
    E_PV_Main_building = np.loadtxt("../../../Hydrogen/PV_Main_building.txt")
    E_agriPV_scenario_north = np.loadtxt("../../../Hydrogen/PV_agriPV_north_east-west.txt")
    E_PV_Total = np.sum([E_PV_STO, E_PV_TIP, E_PV_Main_building, E_agriPV_scenario_north], axis=0)

    P = np.array([El_demand_2022, E_PV_Total])

    #Change to dictionaries
    spot_price_data_2022_dict = dict(enumerate(spot_price_data_2022))
    spot_price_data_2021_dict = dict(enumerate(spot_price_data_2021))
    ma_loss_rt_dict = dict(enumerate(ma_loss_rt))
    El_demand_2022_dict = dict(enumerate(El_demand_2022))
    El_demand_2021_dict = dict(enumerate(El_demand_2021))
    E_PV_Total_dict = dict(enumerate(E_PV_Total))

    return spot_price_data_2022_dict, spot_price_data_2021_dict, ma_loss_rt_dict, El_demand_2022_dict, El_demand_2021_dict, E_PV_Total_dict, P

spot_price_data_2022_dict, spot_price_data_2021_dict, ma_loss_rt_dict, El_demand_2022_dict, El_demand_2021_dict, E_PV_Total_dict, P = readdata()

# Define the model
model = ConcreteModel()

# Define sets
model.time = Set(initialize=range(8760))

# Define variables
model.P_ex = Var(model.time, within=Reals) # power exported at time t [kW]
model.P_imp = Var(model.time, within=Reals) # power imported at time t [kW]

model.P_curt = Var(model.time, within=NonNegativeReals) # power curtailed at time t [kW]
model.P_e = Var(model.time, within=NonNegativeReals) # power to electrolyzer at time t (electrolyzer power) [kW]
model.P_f = Var(model.time, within=NonNegativeReals) # power produced by fuel cell at time t (fuel cell power) [kW]
#model.P_pv = Var(model.time, within=NonNegativeReals) #solar PV power at time t [kW]
#model.P_bal = Var(model.time, within=Reals) # Local power imbalance (PV prod minus electrical load) [kW]
model.P_ns = Var(model.time, within=Reals) # Local consumption not supplied [kW]

model.VH = Var(model.time, within=NonNegativeReals) # Hydrogen storage volume at time t [Nm3]
model.FHnp = Var(model.time, within=NonNegativeReals) # Non-productive hydrogen flow
model.FHe = Var(model.time, within=NonNegativeReals) # Electrolyzer hydrogen flow
model.FHf = Var(model.time, within=NonNegativeReals) # Fuel cell hydrogen flow

#Define parameters
model.P_bal = Param(initialize=0, mutable=True)
#model.VH = Param(initialize=0, mutable=True)

#LHV, efficiencies and specific power consumption/generation
model.LHV_H2 = Param(initialize=3, mutable=False) # Lower Heating Value of H2 - 3 kWh/Nm3
#model.VH_0 = Param(within=PositiveReals, mutable=True) # Initial hydrogen storage level [Nm3]

model.eta_e = Param(initialize=70, mutable=False) # Efficiency of the electrolyzer = 70%
model.eta_f = Param(initialize=53.5, mutable=False) # Efficiency of the fuel cell = 53.5%
model.spc_e = Param(initialize=4.2785714, mutable=True) #Specific power consumption of electrolyzer [kWh/Nm3]
model.spc_f = Param(initialize=1.4975, mutable=True) #Specific power generation in  fuel cell [kWh/Nm4]
model.spc_e_ref = Param(within=PositiveReals, mutable=True) #refrence value for power consumption
model.spg_f_ref = Param(within=PositiveReals, mutable=True) #refrence value for power generation
model.ispg_e = Param(within=PositiveReals, mutable=True) #inverse of spc_e
model.ispg_f = Param(within=PositiveReals, mutable=True) #inverse of spc_f

#Capacity parameters
model.P_exp_max = Param(initialize=100, mutable=False) # grid export capacity, 100 kW
model.P_imp_max = Param(initialize=10000, mutable=False) # grid import capacity, 10 000 kW
model.VH_max = Param(initialize=3400, mutable=False) # H2 storage capacity [Nm3]
model.rat_VH_0 = Param(initialize=50, mutable=False) # 50% of max capacity
model.VH_0 = model.VH_max * model.rat_VH_0 * 0.01
model.VH_secur = model.VH_max * model.rat_VH_0 * 0.01

model.Pe_max = Param(initialize=600, mutable=False) # electrolyzer max allowed capacity [kW]
model.Pf_max = Param(initialize=600, mutable=False) # fuel cell max allowed capacity [kW]

#Financial parameters
model.spot_price = Param(spot_price_data_2022_dict.keys(), initialize=spot_price_data_2022_dict, mutable=False) #spot price at time t [NOK/kWh]
model.ma_loss_rt = Param(ma_loss_rt_dict.keys(), initialize=ma_loss_rt_dict, mutable=False) # marginal loss rate at time t [0.06 or 0.04 (6% or 4%)]

#Solar PV and electricity load
model.P_pv = Param(E_PV_Total_dict.keys(), initialize=E_PV_Total_dict, mutable=False) # power production from all PV systems simulated in PVsyst [kW]
model.P_l = Param(El_demand_2022_dict.keys(), initialize=El_demand_2022_dict, mutable=False) # hourly load at Skjetlein [kWh]

#Simulation parameters
model.dt = Param(initialize=1, mutable=True) # Time step
model.idt = 1/model.dt #inverse of time step
model.epsilon = Param(initialize=0.001, mutable=False) # Small value

#Printing
#print("Spot price length; ", len(model.spot_price))
#print("ma_loss_rt length; ", len(model.ma_loss_rt))
#print("E_PV_Total length; ", len(model.E_PV_Total))

#for i in range(0,len(model.spot_price)):
#    print(model.spot_price[i])

# Define the objective function
def obj_func(model):
    return sum(model.P_ex[t] * model.spot_price[t] * model.ma_loss_rt[t] for t in model.time)
model.obj = Objective(rule=obj_func, sense=maximize)

#Define constraints
def exp_rule(model, t):
    return model.P_ex[t] <= model.P_exp_max # Plus costumer rule
model.exp_cons = Constraint(model.time, rule=exp_rule)

def h2_rule(model, t):
    return model.P_e[t] <= model.Pe_max # Electrolyzer rule
model.h2_cons = Constraint(model.time, rule=h2_rule)

def FC_rule(model, t):
    return model.P_f[t] <= model.Pf_max # Fuel cell rule
model.FC_cons = Constraint(model.time, rule=FC_rule)

#JESS, DENNE FUNKER!!! I MORGEN MÅ DU PRØVE Å FIKSE FUEL CELL
def electrolyzer_rule(model, t):
    Pe_max = min(model.Pe_max, model.P_exp_max + (model.P_pv[t] - model.P_l[t]))
    Pe_max = max(Pe_max, 0)

    model.spc_e = (100*model.LHV_H2)/model.eta_e
    model.ispc_e = 1/model.spc_e
    model.spg_f = (100*model.LHV_H2)/model.eta_f

    if value(model.P_pv[t]) >= value(model.P_l[t]): # Run Electrolyzer
        model.P_f[t] = 0 #do not run fuel cell
        model.FHf[t] = 0
        model.FHnp[t] = 0

        model.P_e[t] = min(Pe_max, (model.P_pv[t] - model.P_l[t]))
        model.FHe[t] = (1/model.spc_e)*model.P_e[t]

        model.VH[t] = model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t]) #H2 storage level
        secur_limit = model.VH_secur - model.epsilon
        max_limit = model.VH_max + model.epsilon

        if value(model.VH[t]) < secur_limit: # Increase hydrogen production (below security limit)
            model.FHe[t] = (model.VH_secur - model.VH_0) * model.idt
            model.P_e[t] = min(Pe_max, model.spc_e * model.FHe[t])
            model.FHe[t] = model.ispc_e * model.P_e[t]

            model.VH[t] == model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t]) #Update H2 storage level

        elif value(model.VH[t]) > max_limit: # Decrease hydrogen production (above max)
            model.FHe[t] = (model.VH_max - model.VH_0) * model.idt + model.FHl[t]
            model.P_e[t] = min(Pe_max, model.spc_e * model.FHe[t])
            model.FHe[t] = model.ispc_e * model.P_e[t]
    
    return (model.VH[t] == model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t])) #H2 storage level
model.electrolyzer_rule_cons = Constraint(model.time, rule=electrolyzer_rule)

#write a constraint for fuel cell rule

'''
def energy_balance(model, t):
    model.VH_0 = model.VH_max * model.rat_VH_0 * 0.01

    #model.P_pv[t] = model.E_PV_Total[t]
    #model.P_l[t] = model.El_demand[t]
    #model.FHl[t] = p[2]

    model.P_bal = model.P_pv[t] - model.P_l[t] # Solar power minus load
    Pbal = model.P_bal

    Pe_max = min(model.Pe_max, model.P_exp_max + model.P_bal)
    Pe_max = max(Pe_max, 0)

    model.spc_e_ref = model.LHV_H2
    model.spc_e = (100*model.spc_e_ref)/model.eta_e
    model.ispc_e = 1/model.spc_e
    model.spg_f_ref = model.LHV_H2
    model.spg_f = (100*model.spg_f_ref)/model.eta_f

    if Pbal >= 0: #Run Electrolyzer
        model.P_f[t] = 0 #do not run fuel cell
        model.FHf[t] = 0
        model.FHnp[t] = 0

        model.P_e[t] = min(Pe_max, model.P_bal)
        model.FHe[t] = (1/model.spc_e)*model.P_e[t]

        model.VH = model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t]) #H2 storage level
        VH = model.VH

        if VH < (model.VH_secur - model.epsilon): # Increase hydrogen production (below security limit)
            model.FHe[t] = (model.VH_secur - model.VH_0) * model.idt
            model.P_e[t] = min(Pe_max, model.spc_e * model.FHe[t])
            model.FHe[t] = model.ispc_e * model.P_e[t]

            return model.VH == model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t]) #Update H2 storage level

        elif VH > (model.VH_max + model.epsilon): # Decrease hydrogen production (above max)
            model.FHe[t] = (model.VH_max - model.VH_0) * model.idt + model.FHl[t]
            model.P_e[t] = min(Pe_max, model.spc_e * model.FHe[t])
            model.FHe[t] = model.ispc_e * model.P_e[t]

            return model.VH == model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t]) #Update H2 storage level

    else: # Run fuel cell
        model.P_e[t] = 0 # do not run electrolyzer
        model.FHe[t] = 0
        model.FHnp[t] = 0

        model.P_f[t] = min(model.Pf_max[t], -model.P_bal)
        model.FHf[t] = model.ispg_f * model.P_f[t]

        model.VH = model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t]) #H2 storage level
        VH = model.VH

        if VH < (model.VH_secur - model.epsilon): # Increase fuel cell power
            model.FHf[t] = (model.VH_secur - model.VH_0)*model.idt
            model.P_f[t] = max(0, model.spg_f*model.FHf[t])
            model.FHf[t] = model.ispg_f*model.P_f[t]

            model.VH = model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t]) #Update H2 storage level
            VH = model.VH

            if VH < (model.VH_secur - model.epsilon): # Increase hydrogen production (below security limit)
                model.FHe[t] = (model.VH_secur - model.VH_0) * model.idt
                model.P_e[t] = min(Pe_max, model.spc_e * model.FHe[t])
                model.FHe[t] = model.ispc_e * model.P_e[t]
                
                return model.VH == model.VH_0 + model.dt*(model.FHe[t] - model.FHf[t]) #Update H2 storage level
model.energy_balance_cons = Constraint(model.time, rule=energy_balance)

'''

    # Check H2 storage
    #if model.VH[t] < (0 - model.epsilon):
        # Check hydrogen storage
        #model.FHnp[t] = (0 - model.VH[t])*model.idt
        #model.VH[t] = 0
        
    # Updated power balance and hydrogen balance
    #model.P_bal[t] = model.P_pv[t] - model.P_l[t] + model.P_f[t] - model.P_e[t]
    #model.FHout[t] = model.FHl[t] - model.FHnp[t]

    #model.P_ns[t] = 0

    #if model.P_bal[t] > 0: # 
        # Export to grid and/or power dumping (curtailed power)
        #model.P_curt[t] = max(0, model.cleat] - model.P_exp_max)
        #model.P_g[t] = min(model.P_exp_max, model.P_bal[t])
    #else:
        # Import from grid or local balance achieved
        #model.P_g[t] = model.P_bal[t]
        #model.P_curt[t] = 0
        
        #if - model.P_g[t] > (model.P_imp_max + model.epsilon):
            # Electrical energy not supplied
            #model.P_ns[t] = - model.P_g[t] - model.P_imp_max
            #model.P_g[t] = - model.P_imp_max

    #if (model.P_e[t] > (0 + model.epsilon)) and (model.P_f[t] > (0 + model.epsilon)):
    #    raise Exception('Cannot operate fuel cell and electrolyzer at the same time!')

    #if (model.P_curt[t] > (0 + model.epsilon)) and (model.P_f[t] > (0 + model.epsilon)):
    #    raise Exception('Cannot dump power when operating fuel cell!')

    #if (model.FHnp[t] > (0 + model.epsilon)) and (model.P_f[t] > (0 + model.epsilon)):
    #    raise Exception('Cannot dump hydrogen when operating fuel cell!')

    #if (model.VH[t] < (model.VH_secur - model.epsilon)) and (model.P_f[t] > (0 + model.epsilon)):
    #    raise Exception('Cannot operate fuel cell if VH < VH_secur!')

    #Pcheck = model.P_pv[t] + model.P_f[t] - model.P_curt[t] - model.P_e[t] - model.P_l[t] - model.P_g[t] + model.P_ns[t]

    #if (Pcheck > model.epsilon) or (Pcheck < - model.epsilon):
    #    raise Exception('Power balance not fulfilled!')

    #VHcheck = model.ispc_e*model.P_e[t]*model.dt - model.ispg_f*model.P_f[t]*model.dt - model.FHout[t]*model.dt + model.VH_0 - model.VH[t]

    #if (VHcheck > model.epsilon) or (VHcheck < - model.epsilon):
        #raise Exception('Hydrogen balance not fulfilled!')
    
    #x = [model.VH[t]]
    #u = [model.P_e[t], model.P_f[t], model.FHout[t]]
    #w = [model.P_curt[t], model.P_g[t], model.FHnp[t], model.P_ns[t]]

'''
def simloop(model, t, P):
    T = 8760  # Number of timesteps
    dt = 1

    # Initialize X, U and W
    X = np.zeros((T, 1)) # state variables (hydrogen storage level VH)
    U = np.zeros((T, 3)) # control variables (Pe, Pf, Hout)
    W = np.zeros((T, 4)) # dependent variables (Pd, Pg, FHnp, Pnp)

    # Initial storage level
    x0 = model.VH_max * model.rat_VH_0 * 0.01

    #Collect P_PV and P_l in an array P

    for t in range(T):
        # Parameters for timestep t.
        p = P[t, :]

        # Operate the system in timestep t.
        model = energy_balance(model, x0, p, dt)

        x = [model.VH[t]]
        u = [model.P_e[t], model.P_f[t], model.FHout[t]]
        w = [model.P_curt[t], model.P_g[t], model.FHnp[t], model.P_ns[t]]

        # Store variables for timestep t in matrix.
        X[t, :] = x
        U[t, :] = u
        W[t, :] = w

        # Initial value for state variables at timestep t+1
        x0 = x[0]

    return X, U, W
#model.simloop_cons = Constraint(model.time, P, rule=simloop)
'''

# Solve the optimization problem
opt = SolverFactory('gurobi', solver_io="python")
model.dual = Suffix(direction=Suffix.IMPORT)
results = opt.solve(model, tee=True)
results.write(num=2)

def print_results():
    print('-----------Optimal solution (printed decision variables):-------------')
    for v in model.component_data_objects(Var):
        print(str(v), v.value)
    return model

print_results()