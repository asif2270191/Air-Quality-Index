#!/usr/bin/env python
# coding: utf-8

# # Importoing libraries 

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import accuracy_score,confusion_matrix


# # Reading CSV File

# In[9]:


df=pd.read_csv("data.csv",encoding='unicode_escape')


# # Data Understanding

# In[10]:


#LOADING THE DATASET
df.head()


# In[11]:


#df=df.loc[:,~df.columns.str.contains('^Unnamed')]
#df=df.loc[:,~df.columns.str.contains('^Status')]


# In[12]:


#TO CHECK ROWS/COLUMNS--- AS WE SEE 435742(ROWS) AND 13(COLUMNS)
df.shape


# In[13]:


#CHECKING OVERALL INFORMATION ON THE DATASET
df.info()


# In[14]:


#TO CHECK MISSING VALUES OF COLUMNS IN DATASET
df.isnull().sum()


# In[15]:


#TO CHECK DESCRIPTIVE STATS OF THE NUMERIC VALUES PRESENT IN THE DATA LIKE 
#MEAN,STANDARD DEVIATION,MIN VALUES AND MAX VALUES PRESENT IN THE DATA
df.describe()


# In[16]:


#TO CHECK THE UNIQUE VALUES PRESENT IN THE DATAFRAME
df.nunique()


# In[17]:


#TO CHECK COLUMNS PRESENT IN THE DATASET
df.columns


# # Data Visualization

# In[18]:


#pairplot show us IN scatter plot
sns.pairplot(data=df)


# In[19]:


#COUNT OF VALUES PRESENT IN THE STATE COLUMN
df['state'].value_counts()


# In[20]:


#SHOWS US COUNT OF STATE PRESENT IN THE DATASET
state_counts = df['state'].value_counts()
plt.figure(figsize=(16, 8))
state_counts.plot(kind='bar')
plt.xlabel('Frequencies')
plt.ylabel('State')
plt.title('Frequency of State Names')
plt.show()


# In[21]:


#COUNT OF VALUES PRESENT IN THE type COLUMN
df['type'].value_counts()


# In[22]:


#COUNT OF TYPES PRESENT IN THE DATASET
plt.figure(figsize=(16,8))
plt.xticks(rotation=90)
df.type.hist()
plt.xlabel('Type')
plt.ylabel('Frequencies')
plt.show()


# In[23]:


#COUNTS OF VALUES PRESENT IN THE AGENCY COLUMN
df['agency'].value_counts()


# In[24]:


#PLOTING COUNT OF AGENCY PRESENT IN DATASET
plt.figure(figsize=(16,8))
plt.xticks(rotation=90)
df.agency.hist()
plt.xlabel('Agency')
plt.ylabel('Frequencies')
plt.show()


# In[25]:


#TO SHOW NAME OF STATE HAVING  HIGHER So2 LEVELS IN THE AIR WHICH IS UTTRANCHAL FOLLOWED BY UTTRAKHAND
plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.barplot(x='state',y='so2',data=df)
plt.show()


# In[26]:


#SHOWING THE INCREASING ORDER OF THE STATE ON THEIR So2 LEVEL
df[['so2','state']].groupby(['state']).mean().sort_values(by='so2').plot.bar(color='red')
plt.show()


# In[27]:


##TO SHOW NAME OF STATE HAVING  HIGHER No2 LEVELS IN THE AIR WHICH IS WEST BENGAL
plt.figure(figsize=(16,8))
plt.xticks(rotation=90)
sns.barplot(x='state',y='no2',data=df)
plt.show()


# In[28]:


#SHOWING THE INCREASING ORDER OF THE STATE ON THEIR No2 LEVEL
df[['no2','state']].groupby(["state"]).mean().sort_values(by='no2').plot.bar(color='red')
plt.show()


# In[29]:


#DELHI HAS HIGHER RSPM LEVEL COMPARED TO OTHERS STATES
plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.barplot(x='state',y='rspm',data=df)
plt.show()


# In[30]:


#SHOWING THE INCREASING ORDER OF THE STATE ON THEIR RSPM LEVEL
df[['rspm','state']].groupby(["state"]).mean().sort_values(by='rspm').plot.bar(color='red')
plt.show()


# In[31]:


#DELHI HAS HIGHER PM2_5 LEVEL COMPARED TO OTHER STATES
plt.figure(figsize=(16,8))
plt.xticks(rotation=90)
sns.barplot(x='state',y='pm2_5',data=df)
plt.show()


# ## Checking all the null values and treating those null values

# In[32]:


#CHECKING ALL NULL VALUES
nullvalues=df.isnull().sum().sort_values(ascending=False)


# In[33]:


#HIGHER NULL VALUES PRESENT IN PM2_5 AND SPM
nullvalues


# In[34]:


#COUNT(RETURNS NON-NAN VALUE)
null_values_percentage=(df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)


# In[35]:


#CONCATENATING TOTAL NULL VALUES AND THEIR PERCENTAGE OF MISSING VALUES FOR FURTHER IMPUTATION OR COLUMN DELETION
missing_data_with_percentage=pd.concat([nullvalues,null_values_percentage],axis=1,keys=['Total','Percent'])


# In[36]:


#PERCENTAGES OF NULL VALUES PRESENT IN THE DATASET
missing_data_with_percentage


# In[37]:


df.columns


# In[38]:


#DROPPING UNNECESSARY COLUMNS
df.drop(['agency'],axis=1,inplace=True)
df.drop(['stn_code'],axis=1,inplace=True)
df.drop(['date'],axis=1,inplace=True)
df.drop(['sampling_date'],axis=1,inplace=True)
df.drop(['location_monitoring_station'],axis=1,inplace=True)
df.drop(['pm2_5'],axis=1,inplace=True)


# In[39]:


#NOW CHECKING THE NULL VALUES 
df.isnull().sum()


# In[40]:


df.columns


# In[41]:


df


# In[42]:


#
df['location']=df['location'].fillna(df['location'].mode()[0])
df['type']=df['type'].fillna(df['type'].mode()[0])


# In[43]:


df.fillna(0,inplace=True)


# In[44]:


df.columns


# In[45]:


df.isnull().sum()


# In[46]:


df


# In[47]:


df.columns


# # CALCULATE AIR QUALITY INDEX FOR S02 BASED ON FORMULA
# #### The air quality index is a piecewise linear function of the pollutant concentration, At the boundary between AQI categories,there is a discontinuos jump of one AQI unit. To Convert from concentration to AQI this equation is used
# 
# 

# In[48]:


def cal_SOi(so2):
    si=0
    if (so2<=40):
        si= so2*(50/40)

    elif (so2>40 and so2<=80):
        si= 50+(so2-40)*(50/40)

    elif (so2>80 and so2<=380):
        si= 100+(so2-80)*(100/300)

    elif (so2>380 and so2<=800):
        si= 200+(so2-380)*(100/420)

    elif (so2>800 and so2<=1600):
        si= 300+(so2-800)*(100/800)

    elif (so2>1600):
        si= 400+(so2-1600)*(100/800)
    return si


df['SOi']=df['so2'].apply(cal_SOi)
data= df[['so2','SOi']]
data.head(10)


# In[49]:


def cal_Noi(no2):
    ni=0
    if(no2<=40):
     ni= no2*50/40
    elif(no2>40 and no2<=80):
     ni= 50+(no2-40)*(50/40)
    elif(no2>80 and no2<=180):
     ni= 100+(no2-80)*(100/100)
    elif(no2>180 and no2<=280):
     ni= 200+(no2-180)*(100/100)
    elif(no2>280 and no2<=400):
     ni= 300+(no2-280)*(100/120)
    else:
     ni= 400+(no2-400)*(100/120)
    return ni
df['Noi']=df['no2'].apply(cal_Noi)
data= df[['no2','Noi']]
data.head()


# In[50]:


#sub-index of RSPM -(respirable suspended particualte matter concentration)
def cal_RSPMi(rspm):
    rpi=0
    if(rspm<=100):
     rpi = rspm
    elif(rspm>=101 and rspm<=150):
     rpi= 101+(rspm-101)*((200-101)/(150-101))
    elif(rspm>=151 and rspm<=350):
     ni= 201+(rspm-151)*((300-201)/(350-151))
    elif(rspm>=351 and rspm<=420):
     ni= 301+(rspm-351)*((400-301)/(420-351))
    elif(rspm>420):
     ni= 401+(rspm-420)*((500-401)/(420-351))
    return rpi
df['RSPMi']=df['rspm'].apply(cal_RSPMi)
data= df[['rspm','RSPMi']]
data.head(10)


# In[51]:


#sub-index of SPM - suspended particulate matter
def cal_SPMi(spm):
    spi=0
    if(spm<=50):
     spi=spm*50/50
    elif(spm>50 and spm<=100):
     spi=50+(spm-50)*(50/50)
    elif(spm>100 and spm<=250):
     spi= 100+(spm-100)*(100/150)
    elif(spm>250 and spm<=350):
     spi=200+(spm-250)*(100/100)
    elif(spm>350 and spm<=430):
     spi=300+(spm-350)*(100/80)
    else:
     spi=400+(spm-430)*(100/430)
    return spi
   
df['SPMi']=df['spm'].apply(cal_SPMi)
data= df[['spm','SPMi']]
data.head()


# In[52]:


df.columns


# In[53]:


# calculate aqi 
def cal_aqi(si,ni,rspmi,spmi):
    aqi=0
    if(si>ni and si>rspmi and si>spmi):
     aqi=si
    elif(ni>si and ni>rspmi and ni>spmi):
     aqi=ni
    elif(rspmi>si and rspmi>ni and rspmi>spmi):
     aqi=rspmi
    elif(spmi>si and spmi>ni and spmi>rspmi):
     aqi=spmi
    return aqi

df['AQI']=df.apply(lambda x:cal_aqi(x['SOi'],x['Noi'],x['RSPMi'],x['SPMi']),axis=1)
data= df[['state','SOi','Noi','RSPMi','SPMi','AQI']]
data.head()


# In[54]:


def AQI_Range(x):
    if x<=50:
        return "Good"
    elif x>50 and x<=100:
        return "Satisfactory"
    elif x>100 and x<=200:
        return "Moderate"
    elif x>200 and x<=300:
        return "Poor"
    elif x>300 and x<=400:
        return "Very Poor"
    else:
        return "Severe"

df['AQI_Range'] = df['AQI'].apply(AQI_Range)
df.head()


# In[55]:


df['AQI_Range'].value_counts()


# In[ ]:





# In[56]:


#SOi	Noi	RSPMi	SPMi	PMi	AQI	AQI_Range


# # Applying Machine Learning
# ### Splitting the dataset into Dependent and independent columns

# In[57]:


x=df[['SOi','Noi','RSPMi','SPMi']]
y=df['AQI']
x.head()


# In[58]:


y.head()


# In[59]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=70)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# # Linear Regression

# In[60]:


model=LinearRegression()
model.fit(x_train,y_train)


# In[61]:


#predicting_train
train_pred=model.predict(x_train)
#predicting on test
test_pred=model.predict(x_test)


# In[62]:


RMSE_train=(np.sqrt(metrics.mean_squared_error(y_train,train_pred)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(y_test,test_pred)))

print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))

print('-'*50)

print('RSquared value on train:',model.score(x_train,y_train))
print('RSquared value on test:',model.score(x_test,y_test))


# # Decision Tree Regressor

# In[63]:


DT=DecisionTreeRegressor()
DT.fit(x_train,y_train)


# In[64]:


#predicting train
train_preds=DT.predict(x_train)
#predicting on test
test_preds=DT.predict(x_test)


# In[65]:


RMSE_train=(np.sqrt(metrics.mean_squared_error(y_train,train_pred)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(y_test,test_pred)))

print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))

print('-'*50)

print('RSquared value on train:',DT.score(x_train,y_train))
print('RSquared value on test:',DT.score(x_test,y_test))


# # Random Forest Regressor

# In[66]:


RF=RandomForestRegressor().fit(x_train,y_train)


# In[67]:


#predicting_train
train_preds1=RF.predict(x_train)
#predicting om test
test_preds1=RF.predict(x_test)


# In[68]:


RMSE_train=(np.sqrt(metrics.mean_squared_error(y_train,train_preds1)))
RMSE_test=(np.sqrt(metrics.mean_squared_error(y_test,test_preds1)))

print("RMSE TrainingData = ",str(RMSE_train))
print("RMSE TestData = ",str(RMSE_test))

print('-'*50)

print('RSquared value on train:',RF.score(x_train,y_train))
print('RSquared value on test:',RF.score(x_test,y_test))


# # Classification Algorithms

# In[69]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[71]:


#Splitting the data into Dependent and independent columns for classification
X2= df[['SOi','Noi','RSPMi','SPMi']]
Y2=df['AQI_Range']


# In[72]:


#Splitting the data into training and testing data 
X_train2,X_test2,Y_train2,Y_test2=train_test_split(X2,Y2,test_size=0.33,random_state=70)


# ## Logistic Regression

# In[74]:


#fit the model on train data
log_reg=LogisticRegression().fit(X_train2,Y_train2)

#predict on train
train_preds2=log_reg.predict(X_train2)

#accuracy on train
print("Model accuracy on train is:",accuracy_score(Y_train2,train_preds2))

#predict on test 
test_preds2=log_reg.predict(X_test2)

#accuracy on test
print("Model accuracy on test is:",accuracy_score(Y_test2, test_preds2))
print('-'*50)

#kappa score
print('KappaScore is:',metrics.cohen_kappa_score(Y_test2,test_preds2))


# In[76]:


log_reg.predict([[727,327.55,78.2,100]])


# In[78]:


log_reg.predict([[2.7,45,35.16,23]])


# In[79]:


log_reg.predict([[10,2.8,82,20]])


# ## Decision Tree Classifier

# In[81]:


#fit the model on train data
DT2=DecisionTreeClassifier().fit(X_train2,Y_train2)

#predict on train
train_preds3=DT2.predict(X_train2)

#accuracy on train
print('Model accuracy on test is:',accuracy_score(Y_train2,train_preds3))

#predict on test
test_preds3=DT2.predict(X_test2)

#accuracy on test
print('MOdel accuracy on test is :',accuracy_score(Y_test2,test_preds3))
print('-'*50)

#Kappa Score

print('KappaScore is:',metrics.cohen_kappa_score(Y_test2,test_preds3))

    


# ##  Random Forest Classifier

# In[82]:


#fit the model on train data
RF=RandomForestClassifier().fit(X_train2,Y_train2)

#predict on train
train_preds4=RF.predict(X_train2)

#accuracy on train
print('Model accuracy on test is:',accuracy_score(Y_train2,train_preds4))

#predict on test
test_preds4=RF.predict(X_test2)

#accuracy on test
print('Model accuracy on test is :',accuracy_score(Y_test2,test_preds4))
print('-'*50)

#Kappa Score

print('KappaScore is:',metrics.cohen_kappa_score(Y_test2,test_preds4))

    


# ## K-Nearest Neighbours 

# In[83]:


#fit the model on train data
KNN=KNeighborsClassifier().fit(X_train2,Y_train2)

#predict on train
train_preds5=KNN.predict(X_train2)

#accuracy on train
print('Model accuracy on test is:',accuracy_score(Y_train2,train_preds5))

#predict on test
test_preds5=KNN.predict(X_test2)

#accuracy on test
print('Model accuracy on test is :',accuracy_score(Y_test2,test_preds5))
print('-'*50)

#Kappa Score
print('KappaScore is:',metrics.cohen_kappa_score(Y_test2,test_preds5))

    


# In[88]:


#prediction on random values
KNN.predict([[70.4,97.7,178.182,200]])


# In[90]:


#prediction on random values
KNN.predict([[1,2.2,3.12,0]])


# In[91]:


#prediction on random values
KNN.predict([[321,450.2,309.12,199]])


# In[ ]:




