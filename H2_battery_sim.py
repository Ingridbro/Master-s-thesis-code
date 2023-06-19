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
    #Create model
    model = ConcreteModel()
    model.M = Set(initialize=RangeSet(1,24)) # all months (1,..,12), pyomo has 1-indexing
    model.T = Set(initialize=RangeSet(1,17520)) # hours (1,..,17520), 2 years

    #Creating variables
    model.P_g_in = Var(model.T, within=NonNegativeReals) # power from grid to system [kW]
    model.P_g_out = Var(model.T, within=NonNegativeReals) # power from system to grid [kW]
    model.P_electrolyzer = Var(model.T, within=NonNegativeReals)  # power to electrolyzer (charging) [kW]
    model.P_fuel_cell = Var(model.T, within=NonNegativeReals)  # power from fuel cell (discharging) [kW]
    model.P_d = Var(model.T, within=NonNegativeReals) # dumped power (if the battery is full and the PV production exceeds the load) [kW]
    model.P_g_in_max_monthly = Var(model.M, within=NonNegativeReals) # max power from grid to system each month [kW]

    model.H2_cap_max = Param(model.T, initialize=200000, mutable = False) # maximum h2 storage capacity [kWh]
    model.H2_rated_P_max = Param(model.T, initialize=1000, mutable = False) # maximum rated power [kW]

    model.E_bill_before = Var(within=NonNegativeReals) # energy bill before battery + PV [NOK]
    model.P_load_max_monthly = Var(model.M, within=NonNegativeReals) # max load each month [kW]
    
    #H2 Storage variables
    model.H2_st = Var(model.T, within=NonNegativeReals)  # current stored energy in hydrogen storage tank [kWh]
    model.H2_cap = Var(within=NonNegativeReals)  # installed capacity of hydrogen storage [kWh]
    model.H2_rated_P = Var(within=NonNegativeReals)  # rated power of electrolyzer and fuel cell [kW]

    #Constant system parameters
    model.spot_price = Param(model.T, initialize=spot_price_dict) # spot price [NOK/kWh]
    model.spot_peak = Param(model.M, initialize=max_price_dict) # monthly peak spot price [NOK/kWh]
    model.PV = Param(model.T, initialize=PV_dict, mutable=False) # PV production [kW]
    model.ma_loss_rt = Param(model.T, initialize=ma_loss_rt_dict) # marginal loss rate [0.06 or 0.04]

    #grid import/export
    model.P_g_in_max = Param(model.T, initialize=max(load_dict.values()), mutable=False) # grid import capacity - max power from grid to system [kW]
    model.P_g_out_max = Param(model.T, initialize=100, mutable=False) # grid export capacity - max power from system to grid - "Plusskundeordningen" [kW]
    
    model.H2_eof = Param(initialize=0) # end of life percentsge return value, fra masteren
    model.rent = Param(initialize=0.04) # rente, fra masteren

    #Mutable system parameters
    model.P_load = Param(model.T, initialize=load_dict, mutable=False) # electricity demand [kW]
    model.H2_rt_eff = Param(initialize=0.35, mutable=False)  # round trip efficiency of hydrogen storage - https://www.hydrogen.energy.gov/pdfs/review19/sa173_penev_2019_p.pdf
    model.H2_dod = Param(initialize=0.83, mutable=False) # battery depth of discharge (83%) - https://www.hydrogen.energy.gov/pdfs/review19/sa173_penev_2019_p.pdf
    model.H2_OPEX = Param(initialize=223.75, mutable=False)  # hydrogen storage OPEX [NOK/kW/år] - https://www.sciencedirect.com/science/article/pii/S1364032114008284
    model.H2_eic = Param(initialize=39.442, mutable=False) # energy installation cost of hydrogen storage unit [NOK/kWh] - https://pv-magazine-usa.com/2020/07/03/nrel-study-backs-hydrogen-for-long-duration-storage/
    model.H2_pic = Param(initialize=32118, mutable=False) # power installation cost of hydrogen storage unit [NOK/kW] - https://pv-magazine-usa.com/2020/07/03/nrel-study-backs-hydrogen-for-long-duration-storage/
    #model.H2_eic = Param(initialize=40.515, mutable=False)  # energy installation cost of hydrogen storage unit [NOK/kWh] - https://pv-magazine-usa.com/2020/07/03/nrel-study-backs-hydrogen-for-long-duration-storage/
    #model.H2_pic = Param(initialize=32992, mutable=False)  # power installation cost of hydrogen storage unit [NOK/kW] - https://pv-magazine-usa.com/2020/07/03/nrel-study-backs-hydrogen-for-long-duration-storage/
    model.H2_life = Param(initialize=18, mutable=False)  # lifetime of hydrogen storage unit [years] - https://pubs.rsc.org/en/content/articlelanding/2020/ee/d0ee00771d#!divAbstract

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

# Add constraints
def add_constraints(model):
    # Power balance
    def load_flow_rule(model, t):
        return model.P_load[t] == model.P_g_in[t] - model.P_g_out[t] + model.PV[t] - model.P_electrolyzer[t] + model.P_fuel_cell[t] - model.P_d[t]
    model.load_flow_cons = Constraint(model.T, rule=load_flow_rule)

    def energy_storage_tank_rule(model, t):
        if t == 1:
            return model.H2_st[t] == 0
        else:
            return model.H2_st[t] == model.H2_st[t - 1] + (math.sqrt(model.H2_rt_eff) * model.P_electrolyzer[t - 1] - model.P_fuel_cell[t - 1] * (1/math.sqrt(model.H2_rt_eff)))
            
            # se magnus sin PhD - chapter 5 - denne formelen blir brukt der
            # return model.H2_st[t] == model.H2_st[t - 1] + (math.sqrt(model.H2_rt_eff) * model.P_electrolyzer[t - 1] - model.P_fuel_cell[t - 1]*1/math.sqrt(model.H2_rt_eff))
    model.charging_battery_cons = Constraint(model.T, rule=energy_storage_tank_rule)

    #Constraints for charging/discharging
    def charge_storage_rule1(model, t):
        return model.P_electrolyzer[t] <= (1 - model.PV_Discharge_Aux_bin[t]) * (model.PV[t] - model.P_load[t])
    model.charge_storage_cons = Constraint(model.T, rule=charge_storage_rule1)

    def discharge_storage_rule1(model, t):
        return model.P_fuel_cell[t] <= model.PV_Discharge_Aux_bin[t] * (model.P_load[t] - model.PV[t])
    model.discharge_storage_cons = Constraint(model.T, rule=discharge_storage_rule1)

    def discharge_storage_rule2(model, t):
        if t == 1:
            return model.P_fuel_cell[t] == 0
        else:
            return model.P_fuel_cell[t] <= model.H2_st[t-1]
    model.discharge_rule_cons = Constraint(model.T, rule=discharge_storage_rule2)

    def charge_storage_grid_import_rule(model, t): # cannot charge storage from grid
        return model.P_electrolyzer[t] * model.P_g_in[t] == 0
    model.charge_storage_grid_import_cons = Constraint(model.T, rule=charge_storage_grid_import_rule)

    def discharge_storage_grid_export_rule(model, t): # cannot discharge storage to grid
        return model.P_fuel_cell[t] * model.P_g_out[t] == 0
    model.discharge_storage_grid_export_cons = Constraint(model.T, rule=discharge_storage_grid_export_rule)

    def export_and_import(model, t):
        return model.P_g_in[t] * model.P_g_out[t] == 0 # cannot import and export at the same time
    model.export_and_import_cons = Constraint(model.T, rule=export_and_import)

    def storage_tank_rule1(model, t):
        return model.H2_st[t] <= model.H2_cap * model.H2_dod
    model.storage_tank_cons1 = Constraint(model.T, rule=storage_tank_rule1)

    #Limit rules
    def charge_limit(model, t):
        return model.P_electrolyzer[t] <= model.H2_rated_P # cannot charge more than rated power
    model.charge_limit_cons = Constraint(model.T, rule=charge_limit)

    def discharge_limit(model, t):
        return model.P_fuel_cell[t] <= model.H2_rated_P # cannot discharge more than rated power
    model.discharge_limit_cons = Constraint(model.T, rule=discharge_limit)

    #def capacity_rule(model, t):
    #    return model.H2_cap <= model.H2_cap_max[t]
    #model.capacity_cons = Constraint(model.T, rule=capacity_rule)

    #def rated_power_rule(model, t):
    #    return model.H2_rated_P <= model.H2_rated_P_max[t]
    #model.rated_power_cons = Constraint(model.T, rule=rated_power_rule)

    # Define rule for charging hydrogen storage (via electrolyzer)
    #def charge_H2_rule(model, t):
    #    return model.P_electrolyzer[t] <= model.PV[t] # cannot charge more than PV produced at time t
    #model.charge_H2_cons = Constraint(model.T, rule=charge_H2_rule)

    # Define rule for discharging hydrogen storage (via fuel cell)
    #def discharge_H2_rule(model, t):
    #    return model.P_fuel_cell[t] <= model.P_load[t] # cannot discharge more than load at time t
    #model.discharge_H2_cons = Constraint(model.T, rule=discharge_H2_rule)

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

    def P_load_max_monthly_rule(model, t): # max monthly load before pv + battery
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744] # Hours in each month (assuming no leap years)
        accumulated_hours = [sum(hours_per_month[:i]) for i in range(13)]  # Accumulated hours until the beginning of each month
        year = 1 if t <= 8760 else 2  # Determine the year based on the hour
        t_adjusted = t - (8760 * (year - 1))  # Adjust the hour based on the year
        month = next(i for i, hours in enumerate(accumulated_hours) if hours >= t_adjusted)  # Find the current month
        month += 12 * (year - 1)  # Adjust the month based on the year
        return model.P_load_max_monthly[month] >= model.P_load[t]  # Apply the constraint
    model.P_load_max_monthly_cons = Constraint(model.T, rule=P_load_max_monthly_rule)

    # Monthly peak power constraints may be changed for leap years
    def monthly_peak_power_rule(model, t):
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]  # Hours in each month (assuming no leap years)
        accumulated_hours = [sum(hours_per_month[:i]) for i in range(13)]  # Accumulated hours until the beginning of each month
        year = 1 if t <= 8760 else 2  # Determine the year based on the hour
        t_adjusted = t - (8760 * (year - 1))  # Adjust the hour based on the year
        month = next(i for i, hours in enumerate(accumulated_hours) if hours >= t_adjusted)  # Find the current month
        month += 12 * (year - 1)  # Adjust the month based on the year
        return model.P_g_in_max_monthly[month] >= model.P_g_in[t] - model.P_fuel_cell[t] # Apply the constraint
    model.monthly_peak_power_rule_cons = Constraint(model.T, rule=monthly_peak_power_rule)

    return model

def add_objective(model):
    def obj_rule(model):
        Energy_cost = sum((model.spot_price[t] + model.G_ec_hourly[t] + model.G_ct_hourly[t]) * model.P_g_in[t] - (model.spot_price[t] - model.spot_price[t] * model.ma_loss_rt[t]) * model.P_g_out[t] for t in model.T)
        Energy_cost_fixed = 8800 + 4800 # Fixed annual cost of energy 2021 and 2022 - https://ts.tensio.no/kunde/nettleie-priser-og-avtaler/2022-nettleie-bedrift

        #Annualization factor
        epsilon = model.rent/(1-(1+model.rent)**(-model.H2_life))

        #Storage costs
        Investment_cost = (model.H2_eic*model.H2_cap + model.H2_pic*model.H2_rated_P)*epsilon

        #Capacity cost
        Capacity_cost = sum(model.P_g_in_max_monthly[m]*model.G_pt[m] for m in model.M)
        #return Energy_cost + Energy_cost_fixed + Ann_Battery_cost + Capacity_cost
        return Energy_cost + Energy_cost_fixed + Capacity_cost + Investment_cost
        #return Energy_cost + Energy_cost_fixed + Capacity_cost
        #return sum(model.P_d[t] for t in model.T)
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
    P_electrolyzer = [] # power to battery [kW]
    P_fuel_cell = [] # power from battery [kW]
    P_d = [] # curtailment [kW]
    P_g_in_max_monthly = [] # max power from grid to system each month [kW]
    H2_st = [] # current stored energy in battery [kWh]
    H2_cap = [] # installed capacity of battery [kWh]
    H2_rated_P = [] # rated power of battery [kW]
    E_bill_before = [] # energy bill before battery + PV [NOK]
    spot_price = []

    for v in model.component_data_objects(Var):
        s = re.sub(r'\[\d+\]', '', str(v))
        if s == "P_g_in":
            P_g_in.append(v.value)
        elif s == "P_g_out":
            P_g_out.append(v.value)
        elif s == "P_electrolyzer":
            P_electrolyzer.append(v.value)
        elif s == "P_fuel_cell":
            P_fuel_cell.append(v.value)
        elif s == "P_d":
            P_d.append(v.value)
        elif s == "P_g_in_max_monthly":
            P_g_in_max_monthly.append(v.value)
        elif s == "H2_st":
            H2_st.append(v.value)
        elif s == "H2_cap":
            H2_cap.append(v.value)
        elif s == "H2_rated_P":
            H2_rated_P.append(v.value)
        elif s == "E_bill_before":
            E_bill_before.append(v.value)
        
        #print(str(v), v.value) #printing all values
    print('H2_cap (H2 storage capacity): ', H2_cap)
    print('H2_rated_P (rated power of H2 storage system): ', H2_rated_P)
    print('P_curtailed: ', sum(P_d), 'kWh') # total curtailment
    e_pv_total_list = list(e_pv_total_dict.values())
    print('Percentage curtailed: ', (sum(P_d)/sum(e_pv_total_list))*100, '%') # percentage curtailment
    print('P_g_in_max_monthly: ', P_g_in_max_monthly)
    print('E_bill_before: ', E_bill_before)

    return P_g_in, P_g_out, P_electrolyzer, P_fuel_cell, P_d, P_g_in_max_monthly, H2_st, H2_cap, H2_rated_P

def plot_results(P_g_in, P_g_out, P_electrolyzer, P_fuel_cell, P_d, P_g_in_max_monthly, H2_st, e_pv_total_dict):
    print('...plotting time-series data in Python.')
    e_pv_total_list = list(e_pv_total_dict.values()) # changeing dictionary to list
    spot_prices_list = list(spot_price_data_2021_2022_dict.values())  # change dictionary to list

    def plot_time_series(data, label, ylabel, subplot_num):
        df = pd.DataFrame(data, index=time_index, columns=[label])
        plt.subplot(3,1,subplot_num)
        color = 'darkblue' if label == 'Imported power from grid' else \
                'orange' if label == 'PV production' else \
                'darkviolet' if label == 'Exported power to grid' else \
                'blue' if label.startswith('Power to') or label.startswith('Power from') or label == 'Storage state of charge (SoC)' else \
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
    plot_time_series(P_electrolyzer, 'Power to electrolyzer (charging)', 'Power [kW]', 1)
    plot_time_series(P_fuel_cell, 'Power from fuel cell (discharging)', 'Power [kW]', 2)
    plt.suptitle('Charging and discharging power', fontsize=12, y=0.93)
    plt.subplots_adjust(hspace=0.9, bottom=0.2)

    plt.figure(figsize=(12, 15))
    plot_time_series(spot_prices_list, 'Spot price', 'Price [NOK/kWh]', 1)
    plot_time_series(H2_st, 'Storage state of charge (SoC)', 'Energy [kWh]', 2)
    plt.suptitle('Spot price and battery state of charge', fontsize=12, y=0.93)
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
P_g_in, P_g_out, P_electrolyzer, P_fuel_cell, P_d, P_g_in_max_monthly, H2_st, H2_cap, H2_rated_P = print_and_save_results(model)
plot_results(P_g_in, P_g_out, P_electrolyzer, P_fuel_cell, P_d, P_g_in_max_monthly, H2_st, e_pv_total_dict)