import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import lognorm
from scipy.stats import expon, kstest
from scipy.stats import mannwhitneyu
import powerlaw
import re
import sys
sns.set()

def plot_data(data):
    '''
    this function plots the CDF and PDF of the data with Power-law fit
    @params data: pandas Series, the data to be plotted
    @returns: None, plots and shows the CDF and PDF of the data with Power-law fit
    '''
    fit = powerlaw.Fit(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(16,9))

    # Plot PDF of data with power-law fit
    fit.plot_pdf(ax=axes[0], color='blue', linewidth=2, label='Empirical Data')
    fit.power_law.plot_pdf(ax=axes[0], color='orange', linestyle='--', label='Power-law Fit')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('PDF')
    axes[0].legend()
    axes[0].set_title('PDF with Power-law Fit')

    # Plot CDF of data with power-law fit
    fit.plot_cdf(ax=axes[1], color='blue', linewidth=2, label='Empirical Data')
    fit.power_law.plot_cdf(ax=axes[1], color='orange', linestyle='--', label='Power-law Fit')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('CDF')
    axes[1].legend()
    axes[1].set_title('CDF with Power-law Fit')

    plt.tight_layout()
    plt.show()


def fit_powerlaw(data):
    '''
    this function takes a wikipedia page data and fits a 
    power-law distribution to the number of commits per user
    
    @params data: pandas DataFrame with columns 'uerid' and 'timestamp'
    @returns: None, plots and shows the power-law fit along with the KS statistic and p-value
    '''

    commits_per_user = data['userid'].value_counts()

    fit = powerlaw.Fit(commits_per_user)
    ks_statistic = fit.power_law.KS()
    p_value = fit.distribution_compare('power_law', 'exponential')[1]  # bootstrapped p-value
    #ks_statistic, p_value = fit.distribution_compare('power_law', 'exponential')

    #plot 
    plt.figure(figsize=(8,9))
    commits_per_user.plot(kind='hist', bins=100, density=True, alpha=0.5, label="Commit Count")

    fit.power_law.plot_pdf(color='orange', linestyle='--', label="Power-Law Fit")
     
    plt.legend()
    plt.xlabel('Commit Count (Log Scale)')
    plt.ylabel('Frequency (Log Scale)')
    plt.title(f'Commit Counts per User with Power-Law Fit and KS Goodness of Data (Louis Tomlinson)')
    plt.show()


def find_superfans(df):
    '''
    this function takes a dataframe of Wikipedia revision history to 
    - compress revisions to the latest revision per user per day
    - calculate the number of edits per user (daily)
    - identify the top 5% of contributors as superfans
    @params:
        df: DataFrame with 'userid', 'timestamp', and 'text_length' columns
    @returns: 
        pf.DataFrame: compressed_df, A DataFrame with the latest revision per user per day and edit size
        pd.DataFrame: filtered_df, A DataFrame with the latest revision per user for superfans
        pd.DataFrame: superfans, A DataFrame with the top 5% of contributors
    '''
    # covert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    #sort by 'userid', 'date', and 'timestamp' to get the latest revision in each day
    df = df.sort_values(by=['userid', 'date', 'timestamp'])

    #compress revisions to the latest revision per user per day
    compressed_df = df.groupby(['userid', 'date'], as_index=False).last()
    compressed_df = compressed_df.reset_index(drop=True)
    compressed_df = compressed_df.sort_values(by='timestamp')

    #assume first edit is 72 characters, this is true for both of our datasets but may need to be altered for future work
    compressed_df['edit_size'] = compressed_df['text_length'].diff()
    compressed_df['edit_size'].fillna(72, inplace=True)
    compressed_df['timestamp'] = pd.to_datetime(compressed_df['timestamp'])
    compressed_df['day'] = compressed_df['timestamp'].dt.date 

    #calculate number of edits per user per day
    agg = compressed_df.groupby(['day', 'userid']).size().reset_index(name='num_edits')
    superfans = pd.DataFrame()
    superfans['userid'] = agg['userid'].unique()

    #superfan is defined as the top 5% of contributors (agg by day)
    superfans['num_edits'] = superfans['userid'].map(agg.groupby('userid').size())
    superfans['superfan'] = superfans['num_edits'] >= superfans['num_edits'].quantile(0.95)

    compressed_df = compressed_df.merge(superfans[['userid', 'superfan']], on='userid', how='left')

    df_95 = compressed_df['edit_size'].quantile(0.95)

    filtered_df = compressed_df[compressed_df['edit_size'].abs() <= df_95]
    return compressed_df, filtered_df, superfans

def cummulative_edits (data, period='2W'):
    '''
    this function takes a dataframe of Wikipedia revision history and aggregates 
        - the number of revisions per period
    @params:
        data: DataFrame with 'timestamp' column
        period: string, the period to aggregate by, default is 2 weeks
    @returns:
        pd.DataFrame: biweekly_data, A DataFrame with the number of revisions per period and cumulative
        '''
    # Ensure 'timestamp' is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Aggregate revisions by the specified interval
    data[f'{period}_period'] = data['timestamp'].dt.to_period(period).dt.start_time
    biweekly_data = data.groupby(f'{period}_period').size().reset_index(name='revisions')

    # Sort by period and calculate cumulative revisions
    biweekly_data = biweekly_data.sort_values(f'{period}_period')
    biweekly_data['cumulative_revisions'] = biweekly_data['revisions'].cumsum()

    return biweekly_data

def prepare_biweekly_data(data, period='2W'):
    """
    this function prepares biweekly (or specified interval) cumulative revisions data for a given Wikipedia page dataset.

    @params: 
        data (pd.DataFrame): Dataframe containing revision data with at least 'timestamp' column.
        period (str): Frequency for aggregation (default is '2W' for biweekly).
                  Use standard pandas offset aliases (e.g., 'W' for weekly, 'M' for monthly).
    @returns:
        pd.DataFrame: Processed dataframe with cumulative revisions at the specified interval.
    """
    # Ensure 'timestamp' is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Aggregate revisions by the specified interval
    data[f'{period}_period'] = data['timestamp'].dt.to_period(period).dt.start_time
    biweekly_data = data.groupby(f'{period}_period').size().reset_index(name='revisions')

    # Sort by period and calculate cumulative revisions
    biweekly_data = biweekly_data.sort_values(f'{period}_period')
    biweekly_data['cumulative_revisions'] = biweekly_data['revisions'].cumsum()

    return biweekly_data


def includes_string(string, column, df):
    '''
    function to check for if the string exists in a instance of the row we are investigating
    - this function is better for analyzing the column 'comment' since each comment describes 
    what the editor was revising for 
    @params:
        string: string to search for
        column: string, column name to search in
        df: pd.DataFrame, the data to search in
    @returns: 
        pd.DataFrame with a new column 'includes_string' that is 1 if the string is found in the column and 0 otherwise
    '''
    pattern = rf'\[\[.*?\|{string}\]\]'
    df['includes_string'] = df[column].str.contains(pattern, case=False, na=False).astype(int)
    return df


def count_string(string,column,df):
    '''
    function to count the number of times a string appears in a column
    @params:
        string: string to search for
        column: string, column name to search in
        df: pd.DataFrame, the data to search in
    @returns:
        pd.DataFrame with a new column 'count_{string}' that is the number of times the string appears in the column
    '''
    pattern = rf'\b{re.escape(string)}\b'
    df[f'count_{string}'] = df[column].str.count(pattern, flags=re.IGNORECASE)
    return df
