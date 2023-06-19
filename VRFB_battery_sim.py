from pyomo.environ import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import re
import math

def read_data():
    spot_price_data_2021_2022 = np.loadtxt("./data/Nord_Pool_Spot_Price_Trondheim_2021_2022.txt") # Spot price data 2022 and 2021 from NordPool
    ma_loss_rt = np.loadtxt("./data/ma_loss_rt.txt") # Marginal loss rate - 6% during winter (01.11-01.05) and 4% during summer from TENSIO
    el_demand_2021_2022 = np.loadtxt("./data/el_demand_2021_2022.txt") # Electricity demand 2022 and 2021

    e_pv_sto = np.loadtxt("./data/PV_STO.txt") # Simulated PV production from PVSyst (TMY)
    e_pv_tip = np.loadtxt("./data/PV_TIP.txt")
    e_pv_main_building = np.loadtxt("./data/PV_Main_building.txt")
    e_agri_pv_scenario_south = np.loadtxt("./data/PV_agriPV_north_south-north.txt")
    e_pv_total = np.sum([e_pv_sto, e_pv_tip, e_pv_main_building, e_agri_pv_scenario_south], axis=0)

    spot_price_data_2021_2022_dict = dict(enumerate(spot_price_data_2021_2022)) # Change numpy arrays to dictionaries
    ma_loss_rt_dict = dict(enumerate(ma_loss_rt))
    el_demand_2021_2022_dict = dict(enumerate(el_demand_2021_2022))
    e_pv_total_dict = dict(enumerate(e_pv_total))

    return spot_price_data_2021_2022_dict, ma_loss_rt_dict, el_demand_2021_2022_dict, e_pv_total_dict

def create_model(spot_price_dict, PV_dict, load_dict, max_price_dict, ma_loss_rt_dict): #initializing model

    #Create empty model
    model = ConcreteModel()
    model.M = Set(initialize=RangeSet(1,24)) # all months (1,..,12), pyomo has 1-indexing
    model.T = Set(initialize=RangeSet(1,17520)) # hours (1,..,17520), 2 years

    #Creating variables
    model.P_g_in = Var(model.T, within=NonNegativeReals) # power from grid to system [kW]
    model.P_g_out = Var(model.T, within=NonNegativeReals) # power from system to grid [kW]
    model.P_ch = Var(model.T, within=NonNegativeReals) # power to battery [kW]
    model.P_dch = Var(model.T, within=NonNegativeReals) # power from battery [kW]
    model.P_d = Var(model.T, within=NonNegativeReals) # dumped power (if the battery is full and the PV production exceeds the load) [kW]
    model.P_g_in_max_monthly = Var(model.M, within=NonNegativeReals) # max power from grid to system each month [kW]
    
    model.B_cap_max = Param(model.T, initialize=6000, mutable=False) # maximum battery capacity [kWh]
    model.B_rated_P_max = Param(model.T, initialize=500, mutable=False) # maximum rated power of battery [kW]

    model.E_bill_before = Var(within=NonNegativeReals) # energy bill before battery + PV [NOK]
    model.P_load_max_monthly = Var(model.M, within=NonNegativeReals) # max load each month [kW]

    #Battery variables
    model.B_soc = Var(model.T, within=NonNegativeReals) # current stored energy in battery [kWh]
    model.B_cap = Var(within=NonNegativeReals) # installed capacity of battery [kWh]
    model.B_rated_P = Var(within=NonNegativeReals) # rated power of battery [kW]

    #Constant system parameters
    model.spot_price = Param(model.T, initialize=spot_price_dict, mutable=False) # spot price [NOK/kWh]
    model.spot_peak = Param(model.M, initialize=max_price_dict, mutable=False) # monthly peak spot price [NOK/kWh]
    model.PV = Param(model.T, initialize=PV_dict, mutable=False) # PV production [kW]
    model.ma_loss_rt = Param(model.T, initialize=ma_loss_rt_dict, mutable=False) # marginal loss rate [0.06 or 0.04]

    #grid import/export
    model.P_g_in_max = Param(model.T, initialize=max(load_dict.values()), mutable=False) # grid import capacity - max power from grid to system [kW]
    model.P_g_out_max = Param(model.T, initialize=100, mutable=False) # grid export capacity - max power from system to grid - "Plusskundeordningen" [kW]

    model.B_eof = Param(initialize=0) # end of life percentage return value - https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2778219 
    model.rent = Param(initialize=0.04) # rente - https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2778219 

    #Mutable system parameters
    model.P_load = Param(model.T, initialize=load_dict, mutable=False) # electricity demand [kW]
    model.B_rt_eff = Param(initialize=0.8, mutable=False) # battery (DC) round-trip efficiency [%] - https://www.visblue.com/product/container-module
    model.B_dod = Param(initialize=0.9, mutable=False) # battery depth of discharge (100%)
    model.B_soc_max = Param(initialize=0.9, mutable=False) # battery state of charge maximum (90%) (to avoid extreme operating conditions) - https://www.sciencedirect.com/science/article/pii/S2352152X22024896 
    model.B_soc_min = Param(initialize=0.1, mutable=False) # battery state of charge minimum (10%)(to avoid extreme operating conditions) - https://www.sciencedirect.com/science/article/pii/S2352152X22024896 
    model.B_eic = Param(initialize=2306, mutable=False) # energy installation cost of storage unit [NOK/kWh] - IRENA "cost-fo-service" tool
    model.B_pic = Param(initialize=9787, mutable=False) # power installation cost of storage unit [NOK/kW] - IRENA "cost-fo-service" tool
    model.B_life = Param(initialize=20, mutable=False) # battery lifetime [years] - https://www.visblue.com/product/container-module

    #Binary variable
    model.PV_Discharge_Aux_bin = Var(model.T, within=Binary)

    #Economical parameters
    G_ec_dict = {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05, 8: 0.05, 9: 0.05, 10: 0.05, 11: 0.05, 12: 0.05, 13: 0.05, 14: 0.05, 15: 0.05, 16: 0.05, 17: 0.05, 18: 0.05, 19: 0.05, 20: 0.05, 21: 0.05, 22: 0.05, 23: 0.05, 24: 0.05}
    G_pt_dict = {1: 49, 2: 49, 3: 49, 4: 49, 5: 33, 6: 33, 7: 33, 8: 33, 9: 33, 10: 33, 11: 49, 12: 49, 13: 49, 14: 49, 15: 49, 16: 49, 17: 33, 18: 33, 19: 33, 20: 33, 21: 49, 22: 49, 23: 49, 24: 49}
    G_ct_dict = {1: 0.167, 2: 0.167, 3: 0.167, 4: 0.167, 5: 0.167, 6: 0.167, 7: 0.167, 8: 0.167, 9: 0.167, 10: 0.167, 11: 0.167, 12: 0.167, 13: 0.167, 14: 0.167, 15: 0.167, 16: 0.167, 17: 0.167, 18: 0.167, 19: 0.167, 20: 0.167, 21: 0.167, 22: 0.167, 23: 0.167, 24: 0.167}
    # ec = energy cost, pt = power grid tariff, ct = consumption tax
    # Energipris (samme for 2021 og 2022) https://ts.tensio.no/kunde/nettleie-priser-og-avtaler/2021-nettleie-bedrift
    # Gjennomsnitt Nettleie (NMT Effekt lavspent) vinter og sommer (samme for 2021 og 2022) https://ts.tensio.no/kunde/nettleie-priser-og-avtaler/2021-nettleie-bedrift
    # Forbrukeravgift (samme for 2021 og 2022) https://ts.tensio.no/kunde/nettleie-priser-og-avtaler/2021-nettleie-bedrift

    G_ec_hourly_dict = {}
    G_ct_hourly_dict = {}
    for hour in range(0, 2*8760): # two years
        if hour < 8760:  # First year
            month = (datetime.datetime(2021, 1, 1) + datetime.timedelta(hours=hour)).month
            G_ec_hourly_dict[hour + 1] = G_ec_dict[month]
            G_ct_hourly_dict[hour + 1] = G_ct_dict[month]
        else:
            month = (datetime.datetime(2022, 1, 1) + datetime.timedelta(hours=hour-8760)).month
            G_ec_hourly_dict[hour + 1] = G_ec_dict[month]
            G_ct_hourly_dict[hour + 1] = G_ct_dict[month]

    #Grid tariff
    model.G_pt = Param(model.M, initialize=G_pt_dict) # montly peak power coefficient of grid tariff [NOK/kWh]
    model.G_ec_hourly = Param(model.T, initialize=G_ec_hourly_dict) # hourly energy cost of grid tariff [NOK/kWh]
    model.G_ct_hourly = Param(model.T, initialize=G_ct_hourly_dict) # hourly consumption tax of grid tariff [NOK/kWh]
    
    return model

def add_constraints(model):
    # Power balance
    def load_flow_rule(model, t):
        return model.P_load[t] == model.P_g_in[t] - model.P_g_out[t] + model.PV[t] - model.P_ch[t] + model.P_dch[t] - model.P_d[t]
    model.load_flow_cons = Constraint(model.T, rule=load_flow_rule)

    # Current stored kWh
    def energy_storage_rule(model, t):
        if t == 1:
            return model.B_soc[t] == model.B_cap * model.B_soc_min
        else:
            return model.B_soc[t] == model.B_soc[t - 1] + (math.sqrt(model.B_rt_eff) * model.P_ch[t-1] - (1/math.sqrt(model.B_rt_eff)) * model.P_dch[t-1])
    model.charging_battery_cons = Constraint(model.T, rule=energy_storage_rule)

    #Constraints for charging/discharging
    def charge_battery_rule1(model, t):
        return model.P_ch[t] <= (1 - model.PV_Discharge_Aux_bin[t]) * (model.PV[t] - model.P_load[t])
    model.charge_battery_cons = Constraint(model.T, rule=charge_battery_rule1)

    def discharge_battery_rule1(model, t):
        return model.P_dch[t] <= model.PV_Discharge_Aux_bin[t] * (model.P_load[t] - model.PV[t])
    model.discharge_battery_cons = Constraint(model.T, rule=discharge_battery_rule1)

    def discharge_battery_rule2(model, t):
        month = (datetime.datetime(2021, 1, 1) + datetime.timedelta(hours=t)).month
        if t == 1:
            return model.P_dch[t] == 0
        else:
            return model.P_dch[t] <= model.B_soc[t-1]
            #return model.P_dch[t] <= (1 - model.B_sdr)*model.B_soc[t-1]
    model.discharge_rule_cons = Constraint(model.T, rule=discharge_battery_rule2)

    def battery_charge_grid_import(model, t): # battery cannot charge and import from grid at the same time
        return model.P_ch[t] * model.P_g_in[t] == 0
    model.battery_charge_grid_import_cons = Constraint(model.T, rule=battery_charge_grid_import)

    def discharging_and_grid_export(model, t): # USIKKER PÅ DENNE
        return model.P_g_out[t] * model.P_dch[t] == 0 # cannot discharge battery and export to grid at the same time
    model.discharging_and_grid_export_cons = Constraint(model.T, rule=discharging_and_grid_export)

    def export_and_import(model, t):
        return model.P_g_in[t] * model.P_g_out[t] == 0 # cannot import and export at the same time
    model.export_and_import_cons = Constraint(model.T, rule=export_and_import)

    def battery_soc_rule1(model, t):
        return model.B_soc[t] <= model.B_cap * model.B_dod # battery state of charge cannot be higher than the depth of discharge
    model.battery_soc_cons1 = Constraint(model.T, rule=battery_soc_rule1)

    def battery_soc_rule2(model, t):
        return model.B_soc[t] <= model.B_cap * model.B_soc_max # battery state of charge cannot be higher than maximum state of charge
    model.battery_soc_cons2 = Constraint(model.T, rule=battery_soc_rule2)

    def battery_soc_rule3(model, t):
        return model.B_soc[t] >= model.B_cap * model.B_soc_min # battery state of charge cannot be lower than minimum state of charge
    model.battery_soc_cons3 = Constraint(model.T, rule=battery_soc_rule3)

    #Limit rules
    def charge_limit(model, t):
        return model.P_ch[t] <= model.B_rated_P # cannot charge more than rated power
    model.charge_limit_cons = Constraint(model.T, rule=charge_limit)

    def discharge_limit(model, t):
        return model.P_dch[t] <= model.B_rated_P # cannot discharge more than rated power
    model.discharge_limit_cons = Constraint(model.T, rule=discharge_limit)

    def capacity_rule(model, t):
        return model.B_cap <= model.B_cap_max[t] # cannot install more than maximum battery capacity
    model.capacity_cons = Constraint(model.T, rule=capacity_rule)

    def rated_power_rule(model, t):
        return model.B_rated_P <= model.B_rated_P_max[t]
    model.rated_power_cons = Constraint(model.T, rule=rated_power_rule)

    def grid_import_capacity_rule(model, t):
        return model.P_g_in[t] <= model.P_g_in_max[t] # cannot import more than grid import capacity (set to infinite)
    model.grid_import_capacity_cons = Constraint(model.T, rule=grid_import_capacity_rule)

    def grid_export_capacity_rule(model, t):
        return model.P_g_out[t] <= model.P_g_out_max[t] # cannot export more than grid export capacity (set to 100 kW, "Plusskundeordningen")
    model.grid_export_capacity_cons = Constraint(model.T, rule=grid_export_capacity_rule)

    def E_bill_before_rule(model):
        Energy_cost_fixed = 8800 + 4800
        Energy_cost =  sum(model.P_load[t] * (model.spot_price[t] + model.G_ec_hourly[t] + model.G_ct_hourly[t]) for t in model.T)
        Capacity_cost = sum(model.P_load_max_monthly[m] * model.G_pt[m] for m in model.M)
        return model.E_bill_before == Energy_cost + Energy_cost_fixed + Capacity_cost
    model.E_bill_before_cons = Constraint(rule=E_bill_before_rule)

    def P_load_max_monthly_rule(model, t):
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744] # Hours in each month (assuming no leap years)
        accumulated_hours = [sum(hours_per_month[:i]) for i in range(13)]  # Accumulated hours until the beginning of each month
        year = 1 if t <= 8760 else 2  # Determine the year based on the hour
        t_adjusted = t - (8760 * (year - 1))  # Adjust the hour based on the year
        month = next(i for i, hours in enumerate(accumulated_hours) if hours >= t_adjusted)  # Find the current month
        month += 12 * (year - 1)  # Adjust the month based on the year
        return model.P_load_max_monthly[month] >= model.P_load[t]  # Apply the constraint
    model.P_load_max_monthly_cons = Constraint(model.T, rule=P_load_max_monthly_rule)

    def monthly_peak_power_rule(model, t):
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]  # Hours in each month (assuming no leap years)
        accumulated_hours = [sum(hours_per_month[:i]) for i in range(13)]  # Accumulated hours until the beginning of each month
        year = 1 if t <= 8760 else 2  # Determine the year based on the hour
        t_adjusted = t - (8760 * (year - 1))  # Adjust the hour based on the year
        month = next(i for i, hours in enumerate(accumulated_hours) if hours >= t_adjusted)  # Find the current month
        month += 12 * (year - 1)  # Adjust the month based on the year
        #return model.P_g_in_max_monthly[month] >= model.P_g_in[t]  # Apply the constraint
        return model.P_g_in_max_monthly[month] >= model.P_g_in[t] - model.P_dch[t]  # Apply the constraint
    model.monthly_peak_power_rule_cons = Constraint(model.T, rule=monthly_peak_power_rule)

    return model

def add_objective(model):
    def obj_rule(model):
        Energy_cost = sum((model.spot_price[t] + model.G_ec_hourly[t] + model.G_ct_hourly[t]) * model.P_g_in[t] - (model.spot_price[t] - model.spot_price[t] * model.ma_loss_rt[t]) * model.P_g_out[t] for t in model.T)
        Energy_cost_fixed = 8800 + 4800 # Fixed annual cost of energy 2021 and 2022 - https://ts.tensio.no/kunde/nettleie-priser-og-avtaler/2022-nettleie-bedrift

        #Capacity rate
        Capacity_cost = sum(model.P_g_in_max_monthly[m]*model.G_pt[m] for m in model.M)

        #Annualization factor
        epsilon = model.rent/(1-(1+model.rent)**(-model.B_life))
        
        #CAPEX & OPEX
        CAPEX_Investment_cost = (model.B_eic*model.B_cap + model.B_pic*model.B_rated_P)*epsilon
        OPEX_cost = CAPEX_Investment_cost * 0.015

        #return Energy_cost + Energy_cost_fixed + Capacity_cost # control stretegy 1
        return Energy_cost + Energy_cost_fixed + Capacity_cost + CAPEX_Investment_cost + OPEX_cost # Control strategy 2
        #return sum(model.P_d[t] for t in model.T) # maximum sizing for control strategy 1
    model.obj = Objective(rule=obj_rule, sense=minimize)
    return model

def solve_model(model):
    opt = SolverFactory('gurobi', solver_io="python")
    model.dual = Suffix(direction=Suffix.IMPORT)
    results = opt.solve(model, tee=True)
    results.write(num=2)

    return model

def print_and_save_results(model):
    print('-----------Optimal solution (printed decision variables):-------------')
    P_g_in = [] # power from grid to system [kW]
    P_g_out = [] # power from system to grid [kW]
    P_ch = [] # power to battery [kW]
    P_dch = [] # power from battery [kW]
    P_d = [] # curtailment [kW]
    P_g_in_max_monthly = [] # max power from grid to system each month [kW]
    B_soc = [] # current stored energy in battery [kWh]
    B_cap = [] # installed capacity of battery [kWh]
    B_rated_P = [] # rated power of battery [kW]
    E_bill_before = [] # energy bill before battery + PV [NOK]

    for v in model.component_data_objects(Var):
        s = re.sub(r'\[\d+\]', '', str(v))
        if s == "P_g_in":
            P_g_in.append(v.value)
        elif s == "P_g_out":
            P_g_out.append(v.value)
        elif s == "P_ch":
            P_ch.append(v.value)
        elif s == "P_dch":
            P_dch.append(v.value)
        elif s == "P_d":
            P_d.append(v.value)
        elif s == "P_g_in_max_monthly":
            P_g_in_max_monthly.append(v.value)
        elif s == "B_soc":
            B_soc.append(v.value)
        elif s == "B_cap":
            B_cap.append(v.value)
        elif s == "B_rated_P":
            B_rated_P.append(v.value)
        elif s == "E_bill_before":
            E_bill_before.append(v.value)
        
        #print(str(v), v.value) #printing all values
    print('B_cap (battery capacity): ', B_cap)
    print('B_rated_P (rated power of battery): ', B_rated_P)
    print('P_curtailed: ', sum(P_d), 'kWh)') # total curtailment
    print('Percentage curtailed: ', 100*sum(P_d)/sum(e_pv_total_dict.values()), '%') # percentage curtailment
    print('P_g_in_max_monthly: ', P_g_in_max_monthly)
    print('E_bill_before: ', E_bill_before)

    data = {
        'P_load': el_demand_2021_2022_dict.values(),
        'P_pv': e_pv_total_dict.values(),
        'P_g_in': P_g_in,
        'P_g_out': P_g_out,
        'P_ch': P_ch,
        'P_dch': P_dch,
        'P_d': P_d,
        'B_soc': B_soc}

    df = pd.DataFrame(data)
    df.to_csv('model_results.csv', index=False)

    return P_g_in, P_g_out, P_ch, P_dch, P_d, P_g_in_max_monthly, B_soc, B_cap, B_rated_P

def plot_results(P_g_in, P_g_out, P_ch, P_dch, P_d, P_g_in_max_monthly, B_soc, B_cap, B_rated_P):
    print('...plotting time-series data in Python.')
    e_pv_total_list = list(e_pv_total_dict.values())  # change dictionary to list
    spot_prices_list = list(spot_price_data_2021_2022_dict.values())  # change dictionary to list

    def plot_time_series(data, label, ylabel, subplot_num):
        df = pd.DataFrame(data, index=time_index, columns=[label])
        plt.subplot(3, 1, subplot_num)
        color = 'darkblue' if label == 'Imported power from grid' else \
                'orange' if label == 'PV production' else \
                'darkviolet' if label == 'Exported power to grid' else \
                'blue' if label.startswith('Power to') or label == 'Battery state of charge (SoC)' else \
                'black'
        plt.plot(df.index, df[label], label=label, color=color)
        plt.ylabel(ylabel, fontsize=8)
        plt.xlabel('Time [year-month]', fontsize=8)
        plt.gca().tick_params(axis='both', labelsize=8)
        plt.legend(loc='upper left')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.2)
        current_ticks = plt.yticks()[0]
        new_ticks = np.linspace(0, math.ceil(max(current_ticks)), 5)
        plt.yticks(new_ticks)

    # Create a time series for 2022 (hourly data)
    time_index = pd.date_range(start='2021-01-01', end='2022-12-31 23:00:00', freq='H')

    plt.figure(figsize=(12, 15))
    plot_time_series(P_g_in, 'Imported power from grid', 'Power [kW]', 1)
    plot_time_series(P_g_out, 'Exported power to grid', 'Power [kW]', 2)
    plot_time_series(e_pv_total_list, 'PV production', 'Power [kW]', 3)
    plt.suptitle('Grid Power Export and Import', fontsize=12, y=0.93)
    plt.subplots_adjust(hspace=0.9, bottom=0.2)

    plt.figure(figsize=(12, 15))
    plot_time_series(P_ch, 'Power to battery (charging)', 'Power [kW]', 1)
    plot_time_series(P_dch, 'Power to load (discharging)', 'Power [kW]', 2)
    plt.suptitle('Charging and discharging power', fontsize=12, y=0.93)
    plt.subplots_adjust(hspace=0.9, bottom=0.2)

    plt.figure(figsize=(12, 15))
    plot_time_series(spot_prices_list, 'Spot price', 'Price [NOK/kWh]', 1)
    plot_time_series(B_soc, 'Battery state of charge (SoC)', 'Energy [kWh]', 2)
    plt.suptitle('Spot price and battery state of charge', fontsize=12, y=0.93)

    plt.suptitle('Charging and discharging power', fontsize=12, y=0.93)
    plt.subplots_adjust(hspace=0.9, bottom=0.2)

    plt.figure(figsize=(12, 15))
    plot_time_series(P_d, 'Curtailment', 'Power [kW]', 1)
    plt.suptitle('Curtailment', fontsize=12, y=0.93)
    plt.subplots_adjust(hspace=0.9, bottom=0.2)
    plt.show()

spot_price_data_2021_2022_dict, ma_loss_rt_dict, el_demand_2021_2022_dict, e_pv_total_dict = read_data()

def extend_dict(input_dict): # function which extends dictionary to 2 years instead of 1
    n = len(input_dict)
    output_dict = input_dict.copy()
    for i in range(n, 2*n):
        output_dict[i] = input_dict[i-n]
    return output_dict

def remove_negative_values(dictionary): # remove negative spot prices (if there are any)
    new_dict = {}
    for key, value in dictionary.items():
        new_dict[key] = max(value, 0)
    return new_dict

def find_monthly_max_prices(prices_dict): 
    # Initialize the maximum price for each month
    monthly_max_prices = {month: float('-inf') for month in range(1, 25)}

    # Iterate over the hours and update the maximum price for each month
    for hour in range(0, 2*8760): # two years
        price = prices_dict[hour]
        if hour < 8760:  # First year
            month = (datetime.datetime(2021, 1, 1) + datetime.timedelta(hours=hour)).month
        else:  # Second year
            month = (datetime.datetime(2022, 1, 1) + datetime.timedelta(hours=hour-8760)).month
            month = month + 12 # Adjust the month value for the second year
        if price > monthly_max_prices[month]:
            monthly_max_prices[month] = price

    return monthly_max_prices

spot_price_data_2021_2022_dict = remove_negative_values(spot_price_data_2021_2022_dict)
monthly_max_spot_prices = find_monthly_max_prices(spot_price_data_2021_2022_dict)

e_pv_total_dict = extend_dict(e_pv_total_dict)
ma_loss_rt_dict = extend_dict(ma_loss_rt_dict)

# Remove extra data from the dictionaries
spot_price_data_2021_2022_dict.popitem() # remove last hour
spot_price_data_2021_2022_dict.pop(0) # remove first hour

for key, val in e_pv_total_dict.items():
    print(f"Hour {key}: {val}")

def create_new_dict(dictionary): #creating new el_demand with keys from 1-17520 instead of 0-17519
    new_dict = {}
    for key, val in dictionary.items():
        new_key = key + 1
        new_dict[new_key] = val
    return new_dict

new_el_demand_2022_dict = create_new_dict(el_demand_2021_2022_dict)
new_e_pv_total_dict = create_new_dict(e_pv_total_dict)
new_ma_loss_rt_dict = create_new_dict(ma_loss_rt_dict)

model = create_model(spot_price_data_2021_2022_dict, new_e_pv_total_dict, new_el_demand_2022_dict, monthly_max_spot_prices, new_ma_loss_rt_dict)
model = add_constraints(model)
model = add_objective(model)
model = solve_model(model)
P_g_in, P_g_out, P_ch, P_dch, P_d, P_g_in_max_monthly, B_soc, B_cap, B_rated_P = print_and_save_results(model)
plot_results(P_g_in, P_g_out, P_ch, P_dch, P_d, P_g_in_max_monthly, B_soc, B_cap, B_rated_P)