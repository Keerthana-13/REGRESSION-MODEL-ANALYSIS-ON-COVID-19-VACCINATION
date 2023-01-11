#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import hvplot.pandas
import warnings
warnings.filterwarnings('ignore')


# In[111]:


df = pd.read_excel('India_Final.xlsx')
df.tail()


# In[112]:


df.shape


# In[113]:


df.describe


# In[114]:


df.info()


# In[115]:


df_corr = df.corr()
df_corr


# In[116]:


sns.heatmap(df.corr(), annot=True)


# In[117]:


First_Dose= df['Atleast_1st_dose'].unique().sum()
First_Dose


# In[118]:


Fullyvaccinated = df['Fully_vaccinated'].unique().sum()
Fullyvaccinated


# In[119]:


Deaths = df['New_deaths'].unique().sum()
Deaths


# In[120]:


plt.figure(figsize=(10,5),dpi=100)
plt.pie(
    [First_Dose,Fullyvaccinated],
    autopct='%.2F%%',
    labels=['One Dose','Fully_vaccinated'])
plt.title('Ratio of people vaccinated')
plt.show()


# In[121]:


#number of people who got at least one shot of COVID vaccine
df_country_vac = df.groupby('Country').agg({'Atleast_1st_dose':max}).reset_index()
sum_vac = df_country_vac['Atleast_1st_dose'].sum()
df_country_rat = df.groupby('Country').agg({'Ratio':max}).reset_index()
df_country_deaths = df.groupby('Country').agg({'New_deaths':max}).reset_index()
# df_country_deaths
df_country_rat


# In[122]:


#Checking null values
df.isnull()


# In[123]:


#Vaccination vs mortality graph


df_India= df
df_India.plot(
    kind='scatter',
    x='Ratio',
    y='New_deaths',
    label='New_deaths vs ratio(India)',
    alpha=0.2,
    figsize=(20,15),
    s=df_India['New_deaths'],
    c='Ratio',
    cmap=plt.get_cmap('jet'),
    colorbar=True)

plt.xlabel('Ratio(%)',fontsize=15)
plt.ylabel('New_deaths(India)',fontsize=18)
plt.title('ratio vs New_deaths India',fontsize=22)


# In[124]:


sns.pairplot(df)


# In[167]:


df.hvplot.scatter(x='Ratio', y='New_deaths')


# In[127]:


df.hvplot.scatter(x='Fully_vaccinated', y='New_deaths')


# In[128]:


df.hvplot.scatter(x='Fully_vaccinated', y='Ratio')


# In[129]:


df.hvplot.scatter(x='Atleast_1st_dose', y='Fully_vaccinated')


# In[130]:


with plt.style.context(("seaborn", "ggplot")):
    df.plot(
                 x="Ratio",
                 y="New_deaths",
                 kind="scatter",
                 s=100, alpha=0.7,
                 title="vaccination vs Mortality")


# In[131]:


df.hvplot(
                y = ["Atleast_1st_dose", "Fully_vaccinated"],
                width=700, height=400,
                title="Line Chart of All samples of  vaccination")


# In[132]:


df.hvplot(x="Date",y="New_deaths",ylim=(0,3500))


# In[133]:


df = df.drop("Sno", axis=1)
df


# In[134]:


sns.heatmap(df.corr(), annot=True)


# **Splitting into training and testing dataset**

# In[135]:


#By taking 3 independent attributes
X=df[['Atleast_1st_dose','Fully_vaccinated','Ratio']]
y=df['New_deaths']


# In[136]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[137]:


#defining a function for Regression metrics

from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# In[138]:


#Scaling or standardizing the attribute values

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_train
X_test = pipeline.transform(X_test)
X_test


# **Linear Regression**

# In[139]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)


# In[140]:


# print the intercept
print(lin_reg.intercept_)


# In[141]:


coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[142]:


pred = lin_reg.predict(X_test)
pred


# In[143]:


pd.DataFrame({'True Values': y_test, 'Predicted Values': pred}).hvplot.scatter(x='True Values', y='Predicted Values')


# In[144]:


pd.DataFrame({'Error Values': (y_test - pred)}).hvplot.kde()


# In[145]:


test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df


# **Ridge Regression**

# In[146]:


from sklearn.linear_model import Ridge

model= Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred) , cross_val(Ridge())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# **Lasso Regression**

# In[147]:


from sklearn.linear_model import Lasso

model= Lasso(alpha=0.1, 
              precompute=True, 
#               warm_start=True, 
              positive=True, 
              selection='random',
              random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred) , cross_val(Lasso())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# In[148]:


#Graph using alpha
import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0.01,500,100)
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha');


# **Polynomial Regression**

# In[149]:


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

X_train_2_d = poly_reg.fit_transform(X_train)
X_test_2_d = poly_reg.transform(X_test)

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train_2_d,y_train)

test_pred = lin_reg.predict(X_test_2_d)
train_pred = lin_reg.predict(X_train_2_d)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)
results_df_2 = pd.DataFrame(data=[["Polynomial Regression", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)


# In[150]:


from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_test, test_pred) , cross_val(ElasticNet())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# **Robust Regression**

# In[151]:


from sklearn.linear_model import RANSACRegressor

model = RANSACRegressor(base_estimator=LinearRegression(), max_trials=100)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Robust Regression", *evaluate(y_test, test_pred) , cross_val(RANSACRegressor())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# In[152]:


results_df


# In[153]:


#Comparing models using the metric R2 square
results_df.set_index('Model', inplace=True)
results_df['R2 Square'].plot(kind='bar', figsize=(12, 8),title='R2 Square')


# In[109]:


a=results_df['R2 Square']
a


# In[ ]:





# In[154]:


results_df['MSE'].plot(kind='bar', figsize=(12, 8))


# In[155]:


results_df['MAE'].plot(kind='bar', figsize=(12, 8))


# In[156]:


results_df['RMSE'].plot(kind='bar', figsize=(12, 8))


# **Linear and polynomial regression using only one independent attribute**

# In[ ]:





# In[157]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn import linear_model
import pylab as pl


# In[158]:


mydf = df[df.Country == "India"]


# **Linear regression**

# In[159]:


regr = linear_model.LinearRegression()
# We define training variables via Numpy in arrays .
train_x = np.asanyarray(mydf[['Ratio']])
train_y = np.asanyarray(mydf[['New_deaths']])



# Using fit(x, y) method of Scikit-Learn object, we fit the model on the training variables.
regr.fit (train_x, train_y)
# The coefficients and Intercept of this Simple Linear Regression
print(f'Coefficients: {regr.coef_[0][0]}')
print(f'Intercept: {regr.intercept_[0]}')
# Now it is time to draw the line we want using Coefficients and Intercept. 
plt.scatter(mydf.Ratio, mydf.New_deaths,  color='gold') 
XX = train_x
YY = regr.intercept_[0] + regr.coef_[0][0]*train_x   # y = Intercept + (Coeff * VaccinationRate)
# Plotting Regression Line
plt.plot(XX, YY, color='red')
plt.title("India")
plt.xlabel("Vaccination rate (%) ")
plt.ylabel("New deaths")
plt.show()


# In[160]:


# Calculate Predicted values by this model
test_x = np.asanyarray(mydf[['Ratio']])
test_y = np.asanyarray(mydf[['New_deaths']])
predict_y = regr.predict(train_x)
# Using Predicted values to mesure Error of this model
# Mean absolute error
MAE = np.mean(np.absolute(predict_y  - test_y))  
print(f"Mean absolute error: {MAE:.2f}")
# Mean squared error
MSE =  np.mean((predict_y  - test_y) ** 2)
print(f"Residual sum of squares (MSE): {MSE:.2f}")
# R2-score
r2 = r2_score(test_y , predict_y)
print(f"R2-score: {r2:.2f}")


# **Function for polynomial regression with different degress**

# In[161]:


def plot_vaccine_mortality(country_name, df, degree=2):
    
    print(f"{country_name:-^80}")
    # Store country data in a variable 
    mydf = df[df.Country == country_name]
    # Divide data randomly into two test and training sections 
    msk = np.random.rand(len(mydf)) < .8
    train = mydf[msk]
    test = mydf[~msk]
    # Identify the dependent(y) and independent variables(x) in the train dataframe
    train_x = np.asanyarray(train[['Ratio']])
    train_y = np.asanyarray(train[['New_deaths']])
    # Identify the dependent(y) and non-dependent(x) variables in the test dataframe
    test_x = np.asanyarray(train[['Ratio']])
    test_y = np.asanyarray(train[['New_deaths']])
    # Generate polynomial and interaction features Object with our desired degree  
    poly = PolynomialFeatures(degree=degree)
    # In this section, we make a number of variables with different degrees from 
    # independent variables(x) to use them in a multiple regression model.
    train_x_poly = poly.fit_transform(train_x)
    
    
     # Make the model 
    clf = linear_model.LinearRegression()
    train_y_ = clf.fit(train_x_poly, train_y)
    # Print The coefficients
    print ('Coefficients: ')
    for i, c in enumerate(clf.coef_[0]):
        if i: print(f"{c:->22.10f} * X^{i}")
    # Print The Intercept    
    print ('Intercept: ',clf.intercept_[0])
    # Constructing a scatterplot using train data with random color
    plt.scatter(train.Ratio, train.New_deaths,  color= np.random.rand(3,))
    # Set the X axis using numpy:   np.arange(start, end, interval)
    XX = np.arange(train_x[0], train_x[-1], 0.1)
    
    # Set the Y axis using intercept and coefficients that we found in previous steps
    YY = clf.intercept_[0] 
    for d in range(1,degree+1):
        YY += clf.coef_[0][d]*np.power(XX, d)
    # On the previous scatterplot, we fit the regression model with red color. 
    plt.plot(XX, YY, '-r' )
    plt.title(country_name)
    plt.xlabel("Vaccination rate (%) ")
    plt.ylabel("New deaths")
    plt.show()
    # Now it's time to evaluate the model we build 
    # Calculate Predicted values by this model
    test_x_poly = poly.fit_transform(test_x)
    predict_y = clf.predict(test_x_poly)
    # Using Predicted values to mesure Error of this model
    # Mean absolute error
    MAE = np.mean(np.absolute(predict_y - test_y))  
    print(f"Mean absolute error: {MAE:.2f}")
    # Mean squared error
    MSE =  np.mean((predict_y - test_y) ** 2)
    print(f"Residual sum of squares (MSE): {MSE:.2f}")
    
    # R2-score
    r2 = r2_score(test_y, predict_y)
    print(f"R2-score: {r2:.2f}")
    #---------------------------
    print("-"*80)


# In[162]:


#Degree 2
plot_vaccine_mortality("India", df, 2)


# In[163]:


plot_vaccine_mortality("India", df, 4)


# In[164]:


plot_vaccine_mortality("India", df, 6)


# **Simple Linear Regression for prediction**

# In[165]:


# define x,y
x = df[['Ratio']]
y = df[['New_deaths']]

# split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

# Standard Normalization(x)
std_x = StandardScaler()
x_train = std_x.fit_transform(x_train)
x_test = std_x.fit_transform(x_test)

# Standard Normalization(y)
std_y = StandardScaler()
y_train = std_y.fit_transform(y_train)
y_test = std_y.fit_transform(y_test)

# fitting in
liner = LinearRegression()
liner.fit(x_train,y_train)

# coefficients
print ('Coefficients: ', liner.coef_)
print ('Intercept: ',liner.intercept_)

# prediction
y_pre = std_y.inverse_transform(liner.predict(x_test))# inverse to original value
print('The prediction of New Deaths is :\n',y_pre)


# In[166]:


x = df['Ratio']
y = df['New_deaths']

plt.figure(figsize=(10,8),dpi=100)
sns.regplot(x=x, y=y)
plt.xlabel('Ratio(India)',fontsize=18)
plt.ylabel('New Deaths',fontsize=18)
plt.title('Simple Linear Regression',fontsize=20)


# In[ ]:





# In[ ]:




