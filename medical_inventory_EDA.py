import os # To manage software and hardware on the computer.
os.chdir(r'C:\Users\madhu\OneDrive\Desktop\Medical Inventory')
os.getcwd()

import pandas as pd # Used of for Data Manipulation.
import numpy as np # used for numerical calculation.
import matplotlib.pyplot as plt # Used for Data Visuvalization
import seaborn as sns # used for Data Visuvalization
import datetime as dt # handle the timeseries data
import calendar # to handle date type data 

#

# used for missing value imputation.
from sklearn.impute import SimpleImputer

# it is used to make pipeline.
from sklearn.pipeline import make_pipeline,Pipeline


# used for Auto Descripive statistics.
import sweetviz
import dtale

# used for Model Evaluvation.
from sklearn import metrics

# It is used to split the data set into train set and test set.
from sklearn.model_selection import train_test_split

# used for Hyperparameter tuning
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold

# It is used to save the model.
import joblib
import pickle 


from sqlalchemy import create_engine  # used for database connection.

inventory = pd.read_excel(r"C:\Users\madhu\OneDrive\Desktop\Medical Inventory\Medical Inventory Optimaization Dataset.xlsx")

engine = create_engine(f'mysql+pymysql://{"root"}:{"madhu123"}@localhost/{"inventory_db"}')

inventory.to_sql('pharmacy_tbl', con = engine, if_exists= 'replace', index=False, chunksize=1000)

query = 'select * from pharmacy_tbl;'

inventory = pd.read_sql_query(query,engine)


#################################### Type Casting ##########################################

inventory.info()

# 1) The ID varialbe is categorical but it given as Numerical.
# 2) The Dateofbill is also also time series but given as object we need to change.

inventory['Patient_ID'] = inventory['Patient_ID'].astype(str)

inventory['Dateofbill'] = inventory['Dateofbill'].astype('datetime64')

################################### Handling Duplicates #################################
inventory.duplicated().sum()

inventory.drop_duplicates(inplace = True,keep = 'first')

################################### Adding of columns for EDA #################################

inventory['bill_of_month'] = inventory['Dateofbill'].dt.strftime('%B')


inventory['bill_of_quater'] = inventory['Dateofbill'].dt.quarter


################################### Handling missing Values #################################
inventory.isna().sum()

group_cols = ['Typeofsales', 'Specialisation', 'Dept']
imputation_columns = ['Formulation', 'DrugName', 'SubCat', 'SubCat1']



for col in imputation_columns:
    inventory[col] = inventory.groupby(group_cols)[col].apply(
            lambda x: x.fillna(x.mode().iloc[0])if not x.mode().empty else x)
inventory.dropna(inplace=True)
inventory.reset_index(drop = True, inplace = True)

inventory.to_csv(r'C:\Users\madhu\OneDrive\Desktop\Medical Inventory\medical dataset.csv', index=True)

################################### EDA (Descriptive Statistics) ################################

# we are getting monthly qunatity sold.
sns.barplot(x='bill_of_month', y='Quantity', data=inventory,estimator=sum, order=list(calendar.month_name)[1:])

# we are getting monthly qunatity sold for each department.
sns.barplot(x='bill_of_month', y='Quantity', data=inventory,estimator=sum, order=list(calendar.month_name)[1:],hue = 'Dept')

# we are getting monthly qunatity sold for each Formulation.
sns.barplot(x='bill_of_month', y='Quantity', data=inventory,estimator=sum, order=list(calendar.month_name)[1:],hue = 'Formulation')

# we are getting Quartely qunatity sold.
sns.barplot(x= 'bill_of_quater',y = 'Quantity',data = inventory,estimator = sum)

# we are getting Quartely qunatity sold for each Departement.
sns.barplot(x= 'bill_of_quater',y = 'Quantity',data = inventory,estimator = sum,hue = 'Dept')

# we are getting Quartely qunatity sold for each Formulation.
sns.barplot(x= 'bill_of_quater',y = 'Quantity',data = inventory,estimator = sum,hue = 'Formulation')


###########                 ### As like Quantity the same find for the ReturnQunatity. ##########                 #################
sns.barplot(x='bill_of_month', y='ReturnQuantity', data=inventory,estimator=sum, order=list(calendar.month_name)[1:])
            
sns.barplot(x='bill_of_month', y='ReturnQuantity', data=inventory,estimator=sum, order=list(calendar.month_name)[1:],hue = 'Dept')

sns.barplot(x='bill_of_month', y='ReturnQuantity', data=inventory,estimator=sum, order=list(calendar.month_name)[1:],hue = 'Formulation')

sns.barplot(x= 'bill_of_quater',y = 'ReturnQuantity',data = inventory,estimator = sum)

sns.barplot(x= 'bill_of_quater',y = 'ReturnQuantity',data = inventory,estimator = sum, hue = 'Dept')

sns.barplot(x= 'bill_of_quater',y = 'ReturnQuantity',data = inventory,estimator = sum,hue = 'Formulation')
###################################                                                  #####################################


########## Getting deparatments preparing tablets.
departmet1 = inventory[inventory['Dept'] == 'Department1']['DrugName'].unique()
departmet2 = inventory[inventory['Dept'] == 'Department2']['DrugName'].unique()
departmet3 = inventory[inventory['Dept'] == 'Department3']['DrugName'].unique()

# All the three preperining tablets
combined = set(departmet1) & set(departmet2) & set(departmet3)

# Getting only that deparatment3 preparing tablets.
department3_only = [i for i in departmet3 if i not in np.append(departmet1, departmet2)]

# Getting only that deparatment3 preparing tablets.
department1_only = [i for i in departmet1 if i not in np.append(departmet2, departmet3)]

# Getting only that deparatment3 preparing tablets.
department2_only = [i for i in departmet2 if i not in np.append(departmet1, departmet3)]


# The quaterly sales for Department3 only preparing tablets.
sns.barplot(x= 'bill_of_quater',y = 'Quantity',data = inventory[inventory['DrugName'].isin(department3_only)],estimator = sum, hue = 'Dept')

# The quaterly sales for Department1 only preparing tablets.
sns.barplot(x= 'bill_of_quater',y = 'Quantity',data = inventory[inventory['DrugName'].isin(department1_only)],estimator = sum, hue = 'Dept')

# The quaterly sales for Department1 only preparing tablets.
sns.barplot(x= 'bill_of_quater',y = 'Quantity',data = inventory[inventory['DrugName'].isin(department2_only)],estimator = sum, hue = 'Dept')

# Department3 preparing tablets demand.
sns.barplot(x= 'bill_of_quater',y = 'Quantity',data = inventory[inventory['DrugName'].isin(departmet3)],estimator = sum, hue = 'Dept')

# Percentage of returns by each department.
(inventory.groupby('Dept')['ReturnQuantity'].sum()/inventory['ReturnQuantity'].sum()).plot(kind = 'bar')

# Percentage of returns by each department.
(inventory.groupby('Dept')['Quantity'].sum()/inventory['Quantity'].sum()).plot(kind = 'bar')

# highest 5 subcat quantity sold.
subcat_percentage = (inventory.groupby('SubCat')['Quantity'].sum()/inventory['Quantity'].sum()).sort_values(ascending = False)[:5]

# highest 5 subcat1 quantity sold.
subcat1_percentage = (inventory.groupby('SubCat1')['Quantity'].sum()/inventory['Quantity'].sum()).sort_values(ascending = False)[:5]

# highest 5 subcat ReturnQuantity sold.
subcat_return_percentage = (inventory.groupby('SubCat')['ReturnQuantity'].sum()/inventory['ReturnQuantity'].sum()).sort_values(ascending = False)[:5]

# highest 5 drugs Returned.
drug_return_percentage = (inventory.groupby('DrugName')['ReturnQuantity'].sum()/inventory['ReturnQuantity'].sum()).sort_values(ascending = False)[:5]

# highest 5 drugs Sold.
drug_percentage = (inventory.groupby('DrugName')['Quantity'].sum()/inventory['Quantity'].sum()).sort_values(ascending = False)[:5]

# Highest 5 specilisations sold.
specialisation_percentage = (inventory.groupby('Specialisation')['Quantity'].sum()/inventory['Quantity'].sum()).sort_values(ascending = False)[:5]

# Highest 5 specilisations sold.
specialisation_return_percentage = (inventory.groupby('Specialisation')['ReturnQuantity'].sum()/inventory['ReturnQuantity'].sum()).sort_values(ascending = False)[:5]

# Top 10 drugs sold.
top10_slod_drugs = (inventory.groupby('DrugName')['Quantity'].sum()).sort_values(ascending = False)[:10]

################################### Outliers Analysis ################################
# We can perform this based on the situation.
# Outliers checking.
# created the function to see outliers.
def boxplot():
    inventory.plot(kind = 'box' ,subplots = True,figsize = (15,7))
    plt.subplots_adjust(wspace = 0.75)
boxplot()
# Based on the boxplot we decided to do outliers treatment.
'''
for i in (['Quantity','ReturnQuantity','Final_Cost','Final_Sales','RtnMRP']):
    iqr = inventory[i].quantile(0.75) - inventory[i].quantile(0.25)
    lower_limit = inventory[i].quantile(0.25) - (1.5*iqr)
    upper_limit = inventory[i].quantile(0.75) - (1.5*iqr)
    
    inventory[i] = pd.DataFrame(np.where(inventory[i] > upper_limit,upper_limit,
                                         np.where(inventory[i]<lower_limit,lower_limit,inventory[i])))
    



winsor = Winsorizer(capping_method='iqr',fold = 1.5,
           variables=(['Quantity','ReturnQuantity','Final_Cost','Final_Sales','RtnMRP']))
winsor.fit_transform(inventory)'''
################################### Auto EDA ###################################################
# Sweetviz report
report = sweetviz.analyze(inventory)
report.show_html("inventory_management.html")

# Dtale Report
report = dtale.show(inventory)

report.open_browser()

from autoviz.AutoViz_Class import AutoViz_Class
%matplotlib inline
AutoViz_Class().AutoViz(r'C:\Users\madhu\OneDrive\Desktop\Medical Inventory\medical_preprocessed_data.csv')

################################### EDA & Preprocessing Forecasting for Month ###################################################
df = inventory.loc[:, ['Dateofbill','Quantity']]
df['billofmonth'] = df['Dateofbill'].dt.strftime('%B')
inventory_month = df.groupby('billofmonth')['Quantity'].sum().reset_index()

inventory_month.set_index('billofmonth',inplace = True)

inventory_month.to_csv(r'C:\Users\madhu\OneDrive\Desktop\Medical Inventory\df.csv', index=True)


month_order = order= list(calendar.month_name)[1:]
# Convert 'billofmonth' to a categorical variable with the specified order
inventory_month['billofmonth'] = pd.Categorical(inventory_month.index, categories=month_order, ordered=True)

sns.lineplot('billofmonth','Quantity',data = inventory_month)



from statsmodels.graphics.tsaplots import plot_acf
# Plot autocorrelation using statsmodels.graphics.tsaplots.plot_acf
plot_acf(inventory_month['Quantity'], lags=11)  # Assuming monthly data, set lags to 12 for one year
plt.title('Autocorrelation Plot for Quantity in 2022')
plt.xlabel('Lags (Months)')
plt.ylabel('Autocorrelation')
plt.show()
                    ##################### Auto EDA ###############################

df['billofmonth'] = df['Dateofbill'].dt.month

inventory_month = df.groupby('billofmonth')['Quantity'].sum().reset_index()

inventory_month ['billofmonth'] = pd.to_datetime(inventory_month ['billofmonth'], format='%m')

report = dtale.show(inventory_month)

report.open_browser()


################################### Outliers Analysis Monthly Data ################################

# Outliers checking.
# created the function to see outliers
def boxplot():
    inventory_month.plot(kind = 'box' ,subplots = True,figsize = (15,7))
    plt.subplots_adjust(wspace = 0.75)
boxplot()
# Based on the boxplot we decided to do outliers treatment.

# The Outliers are not removed for the monthly Data because the outlier may be will repeat again for the next time also in the reality.
###########################################################################################################################

                         ##################### Preprocessing for Month #####################

data1 = pd.get_dummies(inventory_month.billofmonth)
inventory_month_preprocess = pd.concat([inventory_month , data1] , axis = 1)
data1.columns
################################### EDA & Preprocessing Forecasting for Week ###################################################
inventory.sort_values(by ='Dateofbill',inplace=True)
inventory['weekofbill'] = inventory['Dateofbill'].dt.isocalendar().week
inventory_week = inventory.groupby('weekofbill')['Quantity'].sum().reset_index()


plt.plot(inventory_week.index, inventory_week['Quantity'])
plt.show()

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# Plot autocorrelation using statsmodels.graphics.tsaplots.plot_acf
plot_acf(inventory_week['Quantity'], lags=25) # Assuming monthly data, set lags to 12 for one year
plt.title('Autocorrelation Plot for Quantity in 2022')
plt.xlabel('Lags (Months)')
plt.ylabel('Autocorrelation')
plt.show()
plot_pacf(inventory_week['Quantity'], lags=25)
inventory_week.to_csv(r'C:\Users\madhu\OneDrive\Desktop\Medical Inventory\weekly_inventory.csv', index=True)

sns.lineplot('weekofbill','Quantity',data = inventory_week)


'''
We can now illustrate the problem of time as a cyclical feature. 
We will do so using the weeks of the year as an example. 
Let's extract the weeks from the datetime:
    
Here we see exactly what we would expect from year data for a week: 
    a cycle between 1 and 52 that repeats 52 times for a year.

Encoding Cyclical Features
A common method for encoding cyclical data is to transform the data into two
 dimensions using a sine and consine transformation.

We can do that using the following transformations:

xsin=sin(2∗π∗xmax(x))
xcos=cos(2∗π∗xmax(x))
'''

inventory_week['week_sin'] = np.sin(2*np.pi* inventory_week['weekofbill']/52.0)
inventory_week['week_cos'] = np.cos(2*np.pi* inventory_week['weekofbill']/52.0)


ax = inventory_week['week_sin'].plot()
'''

Great, it appears the absolute difference an week_sin before, 
at and after year is now the same! However, if we look at the plot of week_sin
(following any flat line intersection with the graph), we can see there is a problem. 
If we consider just the one dimension, there are two records with exactly the
same week_sin values.

This is why we also need the cosine transformation,
to separate these records from each other.

'''
################################# Linear Model ############################################

inventory_week.weekofbill = inventory_week.weekofbill.astype('int64')
ax = inventory_week.plot.scatter('week_sin', 'week_cos').set_aspect('equal')

import statsmodels.formula.api as smf 
linear = smf.ols('Quantity ~ weekofbill', data = inventory_week).fit()
linear_predict_week = linear.predict(inventory_week['weekofbill'])

mape_linear = np.mean(np.abs((inventory_week['Quantity'] - linear_predict_week)/inventory_week['Quantity']))*100

np.sqrt(mean_squared_error(linear_predict_week,inventory_week['Quantity']))

# Regression Line
sns.lineplot(inventory_week['weekofbill'], inventory_week['Quantity'])
plt.plot(inventory_week['weekofbill'], linear_predict_week, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

##################################################################################################
################################# Exponetial Model ############################################
inventory_week['log_Quantity'] = np.log(inventory_week['Quantity'])

exponential = smf.ols('log_Quantity ~ weekofbill', data = inventory_week).fit()
exp_predict_week = exponential.predict(inventory_week['weekofbill'])


mape_exp = np.mean(np.abs((inventory_week['Quantity'] - np.exp(exp_predict_week))/inventory_week['Quantity']))*100


sns.lineplot(inventory_week['weekofbill'], inventory_week['Quantity'])
plt.plot(inventory_week['weekofbill'], np.exp(exp_predict_week), "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

##################################################################################################
################################# Quardatic Model ############################################
inventory_week['week_square'] = np.square(inventory_week['weekofbill'])

quardatic = smf.ols('Quantity ~ weekofbill + I(week_square)',data = inventory_week).fit()

quardatic_predict_week = quardatic.predict(inventory_week[['weekofbill','week_square']])

mape_quardatic = np.mean(np.abs((inventory_week['Quantity'] - quardatic_predict_week)
                                /inventory_week['Quantity']))*100


sns.lineplot(inventory_week['weekofbill'], inventory_week['Quantity'])
plt.plot(inventory_week['weekofbill'], quardatic_predict_week, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

##################################################################################################
################################# Additive Seasonality Model ############################################

inventory_week.week_sin = inventory_week.week_sin.astype('float64')
inventory_week.week_cos = inventory_week.week_cos.astype('float64')

add_linear = smf.ols('Quantity ~ week_sin + week_cos',inventory_week).fit()


add_linear_predict_week = add_linear.predict(inventory_week[['week_sin' , 'week_cos']])


mape_add_linear = np.mean(np.abs((inventory_week['Quantity'] - add_linear_predict_week)/inventory_week['Quantity']))*100

##################################################################################################
################################# Multiplicative Seasonality with linear treand ############################################

mul_linear = smf.ols('log_Quantity ~ weekofbill + week_sin + week_cos',data = inventory_week).fit()
mul_linear_predict_week = mul_linear.predict(inventory_week[['week_sin','week_cos','weekofbill']])

mape_mul_linear = np.mean(np.abs((inventory_week['Quantity'] - np.exp(mul_linear_predict_week))/inventory_week['Quantity']))*100

################################# Multiplicative Seasonality with linear treand ############################################

mul_quardatic_linear = smf.ols('log_Quantity ~ weekofbill + week_square + week_sin + week_cos',data = inventory_week).fit()
mul_quardatic_predict_week = mul_quardatic_linear.predict(inventory_week[['week_sin','week_cos','weekofbill','week_square']])

mape_mul_quardatic = np.mean(np.abs((inventory_week['Quantity'] - np.exp(mul_quardatic_predict_week))/inventory_week['Quantity']))*100
