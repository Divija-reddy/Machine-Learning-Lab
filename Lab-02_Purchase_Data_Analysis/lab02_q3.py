# Lab Session 02 - A3
# Statistical Analysis using NumPy
# IRCTC Stock Price Data

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

#  FUNCTION DEFINITIONS 

def load_irctc_data(file_path):
    """
    Loads IRCTC Stock Price worksheet from Excel file
    """
    return pd.read_excel(file_path, sheet_name="IRCTC Stock Price")


def numpy_mean_variance(data):
    """
    Calculates mean and variance using NumPy
    """
    mean_value = np.mean(data)
    variance_value = np.var(data)
    return mean_value, variance_value


def custom_mean(data):
    """
    Calculates mean manually
    """
    total = 0
    for value in data:
        total += value
    return total / len(data)


def custom_variance(data):
    """
    Calculates variance manually
    """
    mean_value = custom_mean(data)
    squared_diff_sum = 0
    for value in data:
        squared_diff_sum += (value - mean_value) ** 2
    return squared_diff_sum / len(data)


def measure_execution_time(function, data, runs=10):
    """
    Measures average execution time of a function
    """
    total_time = 0
    for _ in range(runs):
        start = time.time()
        function(data)
        end = time.time()
        total_time += (end - start)
    return total_time / runs


def filter_wednesday_data(dataframe):
    """
    Filters price data for Wednesdays
    """
    return dataframe[dataframe["Day"] == "Wed"]["Price"].values


def filter_april_data(dataframe):
    """
    Filters price data for April month
    """
    return dataframe[dataframe["Month"] == "Apr"]["Price"].values


def probability_of_loss(chg_percent):
    """
    Calculates probability of loss (negative change)
    """
    loss_days = list(filter(lambda x: x < 0, chg_percent))
    return len(loss_days) / len(chg_percent)


def probability_of_profit_on_wednesday(dataframe):
    """
    Calculates probability of profit on Wednesdays
    """
    wednesday_data = dataframe[dataframe["Day"] == "Wed"]
    profit_days = wednesday_data[wednesday_data["Chg%"] > 0]
    return len(profit_days) / len(wednesday_data)


def scatter_plot_chg_vs_day(dataframe):
    """
    Creates scatter plot of Chg% vs Day of Week
    """
    days = dataframe["Day"]
    chg = dataframe["Chg%"]
    plt.scatter(days, chg)
    plt.xlabel("Day of Week")
    plt.ylabel("Change Percentage (Chg%)")
    plt.title("Chg% vs Day of the Week")
    plt.grid(True)
    plt.show()


#  MAIN FUNCTION 

def main():
    file_path = "Purchase Data.xlsx"
    
    # Load IRCTC data
    irctc_data = load_irctc_data(file_path)
    
    price_data = irctc_data["Price"].values
    chg_percent = irctc_data["Chg%"].values
    
    # Mean and Variance using NumPy
    numpy_mean, numpy_variance = numpy_mean_variance(price_data)
    
    # Mean and Variance using custom functions
    custom_mean_value = custom_mean(price_data)
    custom_variance_value = custom_variance(price_data)
    
    # Execution time comparison
    numpy_time = measure_execution_time(np.mean, price_data)
    custom_time = measure_execution_time(custom_mean, price_data)
    
    # Wednesday & April analysis
    wednesday_prices = filter_wednesday_data(irctc_data)
    april_prices = filter_april_data(irctc_data)
    
    wednesday_mean = custom_mean(wednesday_prices)
    april_mean = custom_mean(april_prices)
    
    # Probabilities
    loss_probability = probability_of_loss(chg_percent)
    profit_wed_probability = probability_of_profit_on_wednesday(irctc_data)
    
    # PRINT STATEMENTS (ONLY HERE) 
    
    print("Population Mean (NumPy):", numpy_mean)
    print("Population Variance (NumPy):", numpy_variance)
    
    print("\nCustom Mean:", custom_mean_value)
    print("Custom Variance:", custom_variance_value)
    
    print("\nAverage Execution Time (NumPy Mean):", numpy_time)
    print("Average Execution Time (Custom Mean):", custom_time)
    
    print("\nWednesday Sample Mean:", wednesday_mean)
    print("April Sample Mean:", april_mean)
    
    print("\nProbability of Making a Loss:", loss_probability)
    print("Probability of Making Profit on Wednesday:", profit_wed_probability)
    
    # Scatter plot
    scatter_plot_chg_vs_day(irctc_data)


if __name__ == "__main__":
    main()
