#!/usr/bin/env python
# coding: utf-8

# In[194]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[195]:


df = pd.read_csv(r'C:\Users\91912\Downloads\hotel_bookings.csv')
type(df)


# In[196]:


df.head(3)


# In[197]:


df.isnull().sum()


# In[198]:


df.drop(['agent','company'],axis=1,inplace=True)


# In[199]:


df['country'].value_counts()


# In[200]:


df['country'].fillna(df['country'].value_counts().index[0],inplace=True)


# In[201]:


df.fillna(0,inplace=True)


# In[202]:


df.isnull().sum()


# In[203]:


filter1 = (df['children']==0) & (df['adults']==0) & (df['babies']==0)


# In[204]:


df[filter1]


# In[205]:


data=df[~filter1]


# In[206]:


data.shape


# In[207]:


data[data['is_canceled']==0]['country'].value_counts()/75011


# In[208]:


len(data[data['is_canceled']==0])


# In[209]:


country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns=['country','no_of_guests']
country_wise_data


# In[210]:


import plotly
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)


# In[211]:


import plotly.express as px


# In[212]:


map_guests = px.choropleth(country_wise_data,
                          locations = country_wise_data['country'],
                          color = country_wise_data['no_of_guests'],
                          hover_name = country_wise_data['country'],
                          title = 'home country of guets')


# In[213]:


map_guests.show()


# In[214]:


data2=data[data['is_canceled']==0]


# In[215]:


data2.columns


# In[216]:


plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',y='adr',hue='hotel',data=data2)
plt.xlabel('room type')
plt.ylabel('price (EUR)')
plt.title('Price of room types per night and person')


# In[217]:


data_resort = data[(data['hotel']=='Resort Hotel') & (data['is_canceled']==0)]
data_city = data[(data['hotel']=='City Hotel') & (data['is_canceled']==0)]
data_resort.head(3)


# In[218]:


rush_resort = data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns=['month','no_of_guests'] 
rush_resort 


# In[219]:


rush_city = data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns=['month','no_of_guests'] 
rush_city 


# In[220]:


final_rush=rush_resort.merge(rush_city,on='month')
final_rush.columns=['month','no_of_guests_in_resort','no_of_guests_in_city']
final_rush


# In[221]:


import sort_dataframeby_monthorweek as sd


# In[222]:


final_rush=sd.Sort_Dataframeby_Month(final_rush,'month')


# In[223]:


px.line(data_frame=final_rush,x='month',y=['no_of_guests_in_resort','no_of_guests_in_city'])


# In[224]:


data = sd.Sort_Dataframeby_Month(data,'arrival_date_month')


# In[225]:


sns.barplot(x='arrival_date_month',y='adr',hue='is_canceled',data=data)
plt.xticks(rotation='vertical')
plt.show()


# In[226]:


pd.crosstab(index=data['stays_in_weekend_nights'],columns=data['stays_in_week_nights'])


# In[227]:


def week_function(row):
    feature1 = 'stays_in_weekend_nights'
    feature2 = 'stays_in_week_nights'
    
    if row[feature2]==0 and row[feature1]>0:
        return 'stay_just_weekend'
    
    elif row[feature2]>0 and row[feature1]==0:
        return 'stay_just_weekdays'
    
    elif row[feature2]>0 and row[feature1]>0:
        return 'stays_both_weekdays_weekends'
    
    else:
        return 'undefined_data' 


# In[228]:


data2['weekend_or_weekday']=data2.apply(week_function,axis=1)


# In[229]:


data2.head(2)


# In[230]:


data2['weekend_or_weekday'].value_counts()


# In[231]:


data2=sd.Sort_Dataframeby_Month(data2,'arrival_date_month')
data2.groupby(['arrival_date_month','weekend_or_weekday']).size()


# In[232]:


group_data=data2.groupby(['arrival_date_month','weekend_or_weekday']).size().unstack().reset_index()


# In[233]:


sorted_data=sd.Sort_Dataframeby_Month(group_data,'arrival_date_month')


# In[234]:


sorted_data.set_index('arrival_date_month',inplace=True)


# In[235]:


sorted_data


# In[236]:


sorted_data.plot(kind='bar',stacked=True,figsize=(15,10))


# In[237]:


def family(row):
    if(row['adults']>0 & (row['babies']>0 or row['children']>0)):
       return 1
    else:
       return 0


# In[238]:


data['is_family'] = data.apply(family,axis=1)


# In[239]:


data['total_customer'] = data['adults'] + data['babies'] + data['children']


# In[240]:


data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']


# In[241]:


data.head(3)


# In[242]:


data['deposit_type'].unique()


# In[243]:


dict1={'No Deposit':0,'Non Refund':1,'Refundable':0}


# In[244]:


data['deposit_given']=data['deposit_type'].map(dict1)


# In[245]:


data.drop(columns=['adults','children','babies'],axis=1,inplace=True)


# In[246]:


data.head(6)


# In[247]:


cat_features=[col for col in data.columns if data[col].dtype=='object']
num_features=[col for col in data.columns if data[col].dtype!='object']


# In[248]:


cat_features


# In[249]:


data_cat=data[cat_features]


# In[250]:


num_features


# In[251]:


data[num_features]


# In[252]:


data.groupby(['hotel'])['is_canceled'].mean()


# In[253]:


import warnings 
from warnings import filterwarnings
filterwarnings('ignore')


# In[254]:


data_cat['cancellation']=data['is_canceled']


# In[255]:


data_cat.head()


# In[256]:


cols=data_cat.columns


# In[257]:


cols=cols[0:-1]


# In[258]:


cols


# In[259]:


for col in cols:
    dict2=data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict2)


# In[260]:


data_cat.head(3)


# In[261]:


dataframe=pd.concat([data_cat,data[num_features]],axis=1)


# In[262]:


dataframe


# In[263]:


dataframe.drop(['cancellation'],axis=1,inplace=True)


# In[264]:


dataframe.head(3)


# In[265]:


sns.distplot(dataframe['lead_time'])


# In[266]:


def handle_outlier(col):
    dataframe[col]=np.log1p(dataframe[col])


# In[267]:


handle_outlier('lead_time')


# In[268]:


sns.distplot(dataframe['lead_time'])


# In[269]:


sns.distplot(dataframe['adr'])


# In[270]:


dataframe[dataframe['adr']<0]


# In[271]:


handle_outlier('adr')


# In[272]:


dataframe['adr'].isnull().sum()


# In[273]:


sns.distplot(dataframe['adr'].dropna())


# In[274]:


sns.FacetGrid(data,hue='is_canceled',xlim=(0,500)).map(sns.kdeplot,'lead_time',shade=True).add_legend()


# In[275]:


corr=dataframe.corr()


# In[276]:


corr


# In[277]:


corr['is_canceled'].sort_values(ascending=False)


# In[278]:


corr['is_canceled'].sort_values(ascending=False).index


# In[279]:


features_to_drop=['reservation_status','reservation_status_date','arrival_date_year',
       'arrival_date_week_number', 'is_family', 'stays_in_weekend_nights',
       'arrival_date_day_of_month']


# In[280]:


dataframe.drop(features_to_drop,axis=1,inplace=True)


# In[281]:


dataframe.shape


# In[282]:


dataframe.dropna(inplace=True)


# In[283]:


x=dataframe.drop('is_canceled',axis=1)


# In[284]:


y=dataframe['is_canceled']


# In[285]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[286]:


feature_sel_model=SelectFromModel(Lasso(alpha=(0.005)))


# In[287]:


feature_sel_model.fit(x,y)


# In[288]:


feature_sel_model.get_support()


# In[289]:


cols=x.columns


# In[290]:


selected_feature=cols[feature_sel_model.get_support()]


# In[291]:


selected_feature


# In[292]:


x=x[selected_feature]


# In[293]:


x


# In[294]:


from sklearn.model_selection import train_test_split


# In[295]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


# In[296]:


X_train.shape


# In[297]:


from sklearn.linear_model import LogisticRegression


# In[298]:


logreg=LogisticRegression()


# In[299]:


logreg.fit(X_train,y_train)


# In[300]:


pred=logreg.predict(X_test)


# In[301]:


pred


# In[302]:


from sklearn.metrics import confusion_matrix


# In[303]:


confusion_matrix(y_test,pred)


# In[304]:


from sklearn.metrics import accuracy_score


# In[305]:


accuracy_score(y_test,pred)


# In[306]:


from sklearn.model_selection import cross_val_score


# In[307]:


score=cross_val_score(logreg,x,y,cv=10)


# In[308]:


score


# In[309]:


score.mean()


# In[310]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[311]:


models=[]
models.append(('Logistic Regression',LogisticRegression()))
models.append(('Naive Bayes',GaussianNB()))
models.append(('Random Forest',RandomForestClassifier()))
models.append(('Decision Tree',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))


# In[312]:


for name,model in models:
    print(name) 
    model.fit(X_train,y_train)
    
    predictions=model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(predictions,y_test))
    print('\n')
    
    print (accuracy_score(predictions,y_test))
    print('\n')

