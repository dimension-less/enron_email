
# coding: utf-8

# ### Importing libraries and magics

# In[1]:


import sys
import os
sys.path.append(os.getcwd()+"/tools/")
from tester import test_classifier


# In[2]:


#Importing libraries and magics


import sys

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import re

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler

#from tester import test_classifier
import warnings




# ### Import the file which contain the dataset to our variable

# In[3]:


# Load the dictionary containing the dataset
with open(os.getcwd()+"/final_project_dataset.pkl", "rb") as data_file:
    data_init = pickle.load(data_file)


# ### Converting the dataset from a python dictionary to a pandas dataframe

# In[4]:


#Converting the dataset from a python dictionary to a pandas dataframe
data_df = pd.DataFrame.from_dict(data_init, orient='index')
raw_data = data_df.copy()


# #### Now check the structure of the new data frame to find out how many total number of observation and column are prsent 

# In[5]:


data_df.shape


# #### Print the first 5 values of the data frame 

# In[6]:


data_df.head()


# #### The dataset contains information of 21 features from 146 employees.

# We can see that column have some values as NaN.
# "NaN”s are actually strings so we will replace them with Numpy’s “NaN”s so that we  can count the values which are not NaN
# across the variables (column).

# In[7]:


data_df.replace(to_replace='NaN', value=np.nan, inplace=True)


# In[8]:


data_df.count().sort_values()


#  Above O/P represent the count of the values in each columns of the data frame

#  We can observe that the dataset is quite sparse with some variables like Total Payments and Total Stock Value having values for most of the employees but some others like Loan Advances and Director Fees that we have information for too few employees.

# We want to find out the records in the data frame which have the mising values. From the above observation we can see that POI variable has the value for all the 146 employees.So we can drop this variable from our data frame. 

# ##### We can also remove the email_address field since we cannot use it somehow in the analysis and we will create a temporary copy without the label (POI).

# In[9]:


#dropping 'poi' and 'email_address' variables
data_df = data_df.drop(["email_address"], axis=1)
data_temp = data_df.drop(["poi"], axis=1)
data_temp[data_temp.isnull().all(axis=1)]


# <b>LOCKHART EUGENE E </b> is the employe in the teporary data frame which has all the missing value for all the variable so we can drop this employee records from the original data fram

# In[10]:


data_df = data_df.drop(["LOCKHART EUGENE E"], axis=0)


# #### Next, since some values are related we would like to rearrange the columns in he following order:

# In[11]:


cols = [
    'poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income',
    'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees',
    'total_payments', 'exercised_stock_options', 'restricted_stock',
    'restricted_stock_deferred', 'total_stock_value',
    'from_poi_to_this_person', 'shared_receipt_with_poi', 'to_messages',
    'from_this_person_to_poi', 'from_messages'
]
data_df = data_df[cols]
data_df.head()


# In[12]:


data_df.replace(to_replace=np.NaN, value=0, inplace=True)


# In[13]:


data_df.head()


# #### Now that the features are in the right order, we can examine the statistics of the dataset.

# In[14]:


data_df.describe()


# ## Outlier Investigation

# My first attempt to spot any possible outliers will be visual.
# We will use Seaborn’s pairplot which present in the same time the distribution of the variables and a scatter plot representation of them

# We are using 4 variables ("total_payments", "exercised_stock_options", "restricted_stock", "total_stock_value") from the data set to plot the graph as these variables has the heighest variance.

# In[15]:


sns.pairplot(data=data_df, vars=["total_payments", "exercised_stock_options", "restricted_stock", "total_stock_value"], hue="poi")


# #### There are two datapoints far away from the cluster of the rest. We will use the Total Payments to find them.

# In[16]:


data_df.total_payments.nlargest(2)


# In[17]:


data_df.loc[['TOTAL']]


# #### The first one ‘TOTAL’, is the totals on the Payments Schedule and not a person so it should be removed.
# The second one is not an outlier, it is just the huge payment and stock value o Kenneth Lay. Datapoints like this are not outliers; in fact anomalies like this may lead us to the rest of the POIs. These extreme values lead the rest of the employees to the bottom left corner of the scatterplot. 

# In[18]:


data_df.drop("TOTAL", inplace=True)


# In[19]:


sns.pairplot(data=data_df, vars=["total_payments", "exercised_stock_options", "restricted_stock", "total_stock_value"], hue="poi")


# #### With the “TOTAL” removed the scatter plots are much more uncluttered and we can see some trends on them.
# We can notice also a negative value on Restricted Stock variable, an indication that more outliers may exist.
# We can make a first sanity by checking if the individual values sum with the totals of each category (Total Payments, Total Stock Value).

# In[20]:


print(data_df.sum()[1:11])
print("---")
print("Sum all 'payment' variables:", sum(data_df.sum()[1:10]))


# In[21]:


print(data_df.sum()[11:15])
print("---")
print("Sum all 'stock' variables:", sum(data_df.sum()[11:14]))


# #### We can see that the totals do not match. We need to check each employee by employee data to find the errors.

# Now to find the error we will find the sum of all the variable from column 2-11 and compare with the value of the 11th variable ("Total_payment").And if the value doenst match we will list those emplyees in the new list.
# Same way we will be comparing the sum of all the variables from column 12-15 and cross check with values of column 15("Total Stock").And if the value doesnt match we will list the emplyee detail in the list

# In[22]:


alist = []
for line in data_df.itertuples():
    if sum(line[2:11]) != line[11] or sum(line[12:15]) != line[15]:
        alist.append(line[0])
data_df.loc[alist]


# In[23]:


data_df.loc["BELFER ROBERT", :]


# In[24]:


data_df.loc["BELFER ROBERT", :] = [
    False, 0, 0, 0, 0, -102500, 3285, 0, 0, 102500, 3285, -44093, 0, 44093, 0,
    0, 0, 0, 0, 0
]
data_df.loc["BHATNAGAR SANJAY", :] = [
    False, 0, 0, 0, 0, 0, 137864, 0, 0, 0, 137864, -2604490, 15456290, 2604490,
    15456290, 0, 463, 523, 1, 29
]


# #### Now that we do not have any more outliers we can plot the two aggregated variables, Total Payments and Total Stock Value.

# In[25]:


fig1, ax = plt.subplots()
for poi, data in data_df.groupby(by="poi"):
    ax.plot(data['total_payments'],data['total_stock_value'],'o', label=poi)
ax.legend()
plt.xscale('symlog')
plt.yscale('symlog')
plt.xlabel("Total Payments")
plt.ylabel("Total Stock Value")

plt.show()


# We can see that there are some persons with zero salary or bonus (or both) and none of them is a POI. Since we have a sparse number of POIs it might be beneficial to remove them to have a more dense dataset. We will create a copy of the dataset with the specific persons removed for future evaluation.

# In[26]:


data_nbs = data_df[data_df.salary > 0]
data_nbs = data_nbs[data_nbs.bonus > 0]
data_nbs.shape


# We can notice that the indexes / names in the dataset are in the form Sirname Name Initial. We will search all the indexes using regular expressions and print the indexes that do not follow this pattern.

# In[27]:


for index in data_df.index:
    if re.match('^[A-Z]+\s[A-Z]+(\s[A-Z])?$', index):
        continue
    else:
        print(index)


# In[28]:


data_df.loc[["THE TRAVEL AGENCY IN THE PARK"]]


# There is a “suspicious” index. The THE TRAVEL AGENCY IN THE PARK, isn’t obviously a name of an employee.So we need to drop it from the data set.

# In[29]:


data_df = data_df.drop(["THE TRAVEL AGENCY IN THE PARK"], axis=0)


# #### As a final step in outlier investigation, We need to search for extreme values.The extreme values is an essential information and they should be kept but let’s spot them first.
# We will be using Tukey Fences with 3 IQRs for every single feature.

# In[30]:


def outliers_iqr(dataframe, features):
    result = set()
    for feature in features:
        ys = dataframe[[feature]]
        quartile_1, quartile_3 = np.percentile(ys, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = int(round(quartile_1 - (iqr * 3)))
        upper_bound = int(round(quartile_3 + (iqr * 3)))
        partial_result = list(np.where((ys > upper_bound) | (ys < lower_bound))[0])
        print(feature, len(partial_result))
        result.update(partial_result)
        
    print("-----------------------------------------------------")
    print("")
    print("Total number of records with extreme values: " + str(len(result)))
    
    return list(result)


# In[31]:


cols.remove("poi")
xtr_values =outliers_iqr(data_df, cols)


# In[32]:


a = data_df.loc[:, "poi"].value_counts()
poi_density = a[1]/(a[0]+a[1])
print("POI density on the original dataset: " + str(poi_density))
a = data_df.ix[xtr_values, "poi"].value_counts()
poi_density_xtr = a[1]/(a[0]+a[1])
poi_density_xtr = a[1]/(a[0]+a[1])
print("POI density on the subset with the extreme values: " + str(poi_density_xtr))

print("Difference: " +str((poi_density_xtr - poi_density) / poi_density))



# We see that in the subset of employees with extreme value to at least one variable, there are 28% more POIs than in the general dataset. This justify our intuition that the extreme values are related with being a POI, thus we will not remove them.

# Now that the dataset is clear of outliers we can find the final dimensions and split the labels from the features and have a first scoring as a baseline for the rest of the analysis. I will use the LinearSVC classifier which seems the more appropriate to begin.

# In[33]:


data_df.shape


# In[34]:


data_df.loc[:, "poi"].value_counts()


# In[35]:


def do_split(data):
    X = data.copy()
    #Removing the poi labels and put them in a separate array, transforming it
    #from True / False to 0 / 1
    y = X.pop("poi").astype(int)
    
    return X, y, 


# In[36]:


X, y = do_split(data_nbs)


# In[37]:


#test_classifier() demands the dataset in a dictionary and the features labels
#in a list with 'poi' first.
features = data_df.columns.tolist()
data = data_df.to_dict(orient='index')

test_classifier(LinearSVC(random_state=42), data, features)


# We are interested in the ability of the classifier not to label as Person Of Interest (POI) a person that is not, and also to find all the POIs so the metrics that we are most interested are Precision and Recall. Since we want to maximize both in the same time we will try to maximize the F1 score which can be interpreted as a weighted average of the precision and recall.

# We can see that the initial scores are very low with the LinearSVC classifier being poor in classifying the right persons.

# So now we will apply the transforming features on the variables and add them to the data set.We will find the propotion of each of the variable and try to add them in the data frame and compare the original variable with new transforming feature  variable in the data  set

# ## Optimize Feature Selection/Engineering

# #### In some cases the value of a variable is less important than its proportion to an aggregated value.

# In[38]:


data = data_df.copy()
data.loc[:, "salary_p"] = data.loc[:, "salary"]/data.loc[:, "total_payments"]
data.loc[:, "deferral_payments_p"] = data.loc[:, "deferral_payments"]/data.loc[:, "total_payments"]
data.loc[:, "loan_advances_p"] = data.loc[:, "loan_advances"]/data.loc[:, "total_payments"]
data.loc[:, "bonus_p"] = data.loc[:, "bonus"]/data.loc[:, "total_payments"]
data.loc[:, "deferred_income_p"] = data.loc[:, "deferred_income"]/data.loc[:, "total_payments"]
data.loc[:, "expenses_p"] = data.loc[:, "expenses"]/data.loc[:, "total_payments"]
data.loc[:, "other_p"] = data.loc[:, "other"]/data.loc[:, "total_payments"]
data.loc[:, "long_term_incentive_p"] = data.loc[:, "long_term_incentive"]/data.loc[:, "total_payments"]
data.loc[:, "director_fees_p"] = data.loc[:, "director_fees"]/data.loc[:, "total_payments"]

data.loc[:, "restricted_stock_deferred_p"] = data.loc[:, "restricted_stock_deferred"]/data.loc[:, "total_stock_value"]
data.loc[:, "exercised_stock_options_p"] = data.loc[:, "exercised_stock_options"]/data.loc[:, "total_stock_value"]
data.loc[:, "restricted_stock_p"] = data.loc[:, "restricted_stock"]/data.loc[:, "total_stock_value"]

data.loc[:, "from_poi_to_this_person_p"] = data.loc[:, "from_poi_to_this_person"]/data.loc[:, "to_messages"]
data.loc[:, "shared_receipt_with_poi_p"] = data.loc[:, "shared_receipt_with_poi"]/data.loc[:, "to_messages"]

data.loc[:, "from_this_person_to_poi_p"] = data.loc[:, "from_this_person_to_poi"]/data.loc[:, "from_messages"]
    
data.replace(to_replace=np.NaN, value=0, inplace=True)
data.replace(to_replace=np.inf, value=0, inplace=True)
data.replace(to_replace=-np.inf, value=0, inplace=True)


# ### Now we can plot the importance of the features of the “enriched” dataset by using the same classifier.

# In[39]:


def plot_importance(dataset):
    X, y = do_split(dataset)

    selector = SelectPercentile(percentile=100)
    a = selector.fit(X, y)

    plt.figure(figsize=(12,9))
    sns.barplot(y=X.columns, x=a.scores_)



# In[40]:


plot_importance(data)


# Comparing the newly created features with the original we can see that the proportions of “Long Term Incentive”, “Restricted Stock Deferred” and “From This Person to POI” score higher than the original features. We will keep these and remove the original values. to avoid bias the model towards a specific feature by using both the original value and its proportion.

# In[41]:


#Adding the proportions
data_df.loc[:, "long_term_incentive_p"] = data_df.loc[:, "long_term_incentive"]/data_df.loc[:, "total_payments"]
data_df.loc[:, "restricted_stock_deferred_p"] = data_df.loc[:, "restricted_stock_deferred"]/data_df.loc[:, "total_stock_value"]
data_df.loc[:, "from_this_person_to_poi_p"] = data_df.loc[:, "from_this_person_to_poi"]/data_df.loc[:, "from_messages"]
#Removing the original values.
data_df.drop("long_term_incentive", axis=1)
data_df.drop("restricted_stock_deferred", axis=1)
data_df.drop("from_this_person_to_poi", axis=1)
#Correcting NaN and infinity values created by zero divisions
data_df.replace(to_replace=np.NaN, value=0, inplace=True)
data_df.replace(to_replace=np.inf, value=0, inplace=True)
data_df.replace(to_replace=-np.inf, value=0, inplace=True)


# In[42]:


plot_importance(data_df)


# ### Sequential Exception Technique (SET):
# 
# Identify the POIs using SET and print their names.

# In[43]:


SET_data = data_df.copy()


# In[44]:


cols = [0,11,12,13,14,15,16,17,18,19,20,21,22]
SET_data.drop(SET_data.columns[cols], axis=1, inplace=True)


# In[45]:


SET_data.head()


# In[46]:


cols = [
    'salary', 'bonus', 'long_term_incentive', 'deferred_income',
    'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees',
    'total_payments'
]


# In[47]:


scaler = MinMaxScaler()
SET_data[cols] = scaler.fit_transform(SET_data[cols])


# In[48]:


e_names = pd.Series(SET_data.index)


# In[ ]:


def SET(m,SET_data):
# Set the value of parameter m = the no. of iterations you require
    Card = pd.Series(np.NAN)
    DS=pd.Series(np.NAN)
    idx_added = pd.Series(np.NAN)
    pos = 0
    for j in range(1,m+1):
        new_indices = np.random.choice(e_names.index,len(e_names),replace=False)
        for i in pd.Series(new_indices).index:
            idx_added[i+pos] = new_indices[i]
            DS[i+pos]=sum(np.var(SET_data.loc[e_names[new_indices[:i+1]]]))
            Card[i+pos] = len(e_names[:i+1])
        pos = pos+i+1

    df = pd.DataFrame({'Index_added':idx_added,'DS':DS,'Card':Card})
    df ['DS_Prev'] = df.DS.shift(1)
    df['Card_prev'] = df.Card.shift(1)
    df.Card_prev[(df.Card == 1)] = 0
    df = df.fillna(0)
    df['Smoothing'] = (df.Card - df.Card_prev)*(df.DS - df.DS_Prev)


    # find indexes of sets with max sf
    maxsf = []
    for i in range(len(df.DS)):
        if df.Smoothing[i] == df.Smoothing.max():
            maxsf.append(i)
    #print(maxsf)

    N = len(e_names)
    excp_set = []
    for i in range(len(maxsf)):
        j = maxsf[i]
        k=j+1
        temp = []
        temp.append(df.Index_added[j])
        excp_set.append(temp.copy())
        temp_prev = pd.DataFrame()
        temp_j = pd.DataFrame()
        a=j
        while(a%N!=0):
            temp_row = SET_data.loc[e_names[df.Index_added[a]]]
            temp_j = temp_j.append(temp_row)
            a=a-1
        temp_row = SET_data.loc[e_names[df.Index_added[a]]]
        temp_j = temp_j.append(temp_row)
        temp_prev = temp_j.copy()                   # Ij-1
        temp_prev.drop(temp_prev.index[0],inplace=True)
        #temp_prev.index = np.arange(len(temp_prev))
        while(k%N!=0):
            K_element = SET_data.loc[e_names[df.Index_added[k]]]    # K th element
            temp_prev = temp_prev.append(K_element)            # Ij-1 U {ik}
            temp_j = temp_j.append(K_element)               # Ij U {ik}
            Dk0 = sum(np.var(temp_prev)) - df.DS[j-1]
            Dk1 = sum(np.var(temp_j)) - df.DS[j]
            if Dk0-Dk1 >= df.DS[j]:                # If Dk0 - Dk1 >= Dj we add the element to exception set.
                excp_set[i].append(df.Index_added[k])
            temp_prev.drop(temp_prev.index[len(temp_prev)-1],inplace=True)
            temp_j.drop(temp_j.index[len(temp_j)-1],inplace=True)
            k+=1
    #print(excp_set)                                # contains the indices of exception elements.
    return excp_set


# In[ ]:


excp_set = SET(1000,SET_data)


# In[ ]:


# Printing the POIs.
print("\nException set: \n")
for i in range(len(excp_set)):
    print(e_names[excp_set[i]])

