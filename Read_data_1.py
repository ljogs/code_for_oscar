# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 02:52:35 2022

@author: LJOGS
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,Lasso, ElasticNet

from sklearn.model_selection import train_test_split

import matlab.engine
eng = matlab.engine.start_matlab()

df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

df_Height = df['Height'];

df_Weight = df['Weight'];

df_BMI = df_Weight/df_Height**2

df = df.drop(columns=['Height', 'Weight'])


columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE",
           "SCC", "CALC", "MTRANS", "NObeyesdad"]

for col in columns:
    df[col] = df[col].astype('category')
    

d = {'Male':1,'Female':0}
df['Gender'] = df['Gender'].map(d)

d = {'yes':1,'no':0}
df['family_history_with_overweight'] = df['family_history_with_overweight'].map(d)

d = {'yes':1,'no':0}
df['FAVC'] = df['FAVC'].map(d)

d = {'Always':3,'Frequently':2,'Sometimes':1,'no':0}
df['CAEC'] = df['CAEC'].map(d)

d = {'yes':1,'no':0}
df['SMOKE'] = df['SMOKE'].map(d)

d = {'yes':1,'no':0}
df['SCC'] = df['SCC'].map(d)

d = {'Always':3,'Frequently':2,'Sometimes':1,'no':0}
df['CALC'] = df['CALC'].map(d)

df = df.drop(columns=['NObeyesdad'])

df = pd.get_dummies(df, columns=['MTRANS'])
df = df.drop(columns=['MTRANS_Public_Transportation'])

colums_names = df.columns

reg = LinearRegression().fit(df, df_BMI)

X=np.mat(df);
y=df_BMI;

p = len(df.T);
std_X = [];
for i in range(p):
    std_X.append(np.std(X[:,i]))

std_X = np.array(std_X);
std_Y = np.std(y)

for i in range(p):
    X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
    
y=(y-np.mean(y))/np.std(y)


##########################################################

df = pd.DataFrame(X)

df.rename(columns={'Gender', 'Age', 'family_history_with_overweight', 'FAVC', 'FCVC',
       'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC',
       'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
       'MTRANS_Walking'})

f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
#plt.title('Correlation Matrix for Obesity Data Set', fontsize=16);



X_rest, X_cv1, y_rest, y_cv1 =  train_test_split(X, y, test_size=1/5, random_state=10)

X_rest, X_cv2, y_rest, y_cv2 =  train_test_split(X_rest, y_rest, test_size=1/4, random_state=10)

X_rest, X_cv3, y_rest, y_cv3 =  train_test_split(X_rest, y_rest, test_size=1/3 , random_state=10)

X_cv5, X_cv4, y_cv5, y_cv4 =  train_test_split(X_rest, y_rest, test_size=1/2 , random_state=10)

X_train1 = np.concatenate((X_cv2, X_cv3, X_cv4, X_cv5));

X_train2 = np.concatenate((X_cv1, X_cv3, X_cv4, X_cv5));

X_train3 = np.concatenate((X_cv1,X_cv2, X_cv4, X_cv5));

X_train4 = np.concatenate((X_cv1,X_cv2, X_cv3, X_cv5));

X_train5 = np.concatenate((X_cv1,X_cv2, X_cv3, X_cv4));

y_train1 = np.concatenate((y_cv2, y_cv3, y_cv4, y_cv5));

y_train2 = np.concatenate((y_cv1, y_cv3, y_cv4, y_cv5));

y_train3 = np.concatenate((y_cv1, y_cv2, y_cv4, y_cv5));

y_train4 = np.concatenate((y_cv1, y_cv2, y_cv3, y_cv5));

y_train5 = np.concatenate((y_cv1, y_cv2, y_cv3, y_cv4));


####################################OSCAR###############

def lossf(X, y, coef):
    predictY_test = np.array(np.dot(X, coef))[0];
    return sum(pow(predictY_test-y, 2))


tuning_log=[]
oscar_test_log=[]


####################################OSCAR1###############

OSCAR_X_train = pd.DataFrame(X_train1);
OSCAR_X_test = pd.DataFrame(X_cv1);
OSCAR_y_train = pd.DataFrame(y_train1);
OSCAR_y_test = pd.DataFrame(y_cv1);

OSCAR_X_train.to_csv('OSCAR_X_train.csv', index=False, header=False);
OSCAR_X_test.to_csv('OSCAR_X_test.csv', index=False, header=False);
OSCAR_y_train.to_csv('OSCAR_y_train.csv', index=False, header=False);
OSCAR_y_test.to_csv('OSCAR_y_test.csv', index=False, header=False);

oscar_coef, d_f, tuning= eng.oscar_package_real_data(nargout=3)

oscar_coef=np.array(oscar_coef)
oscar_coef=oscar_coef[:,0];
tuning_log.append(tuning);

oscar_test = lossf(X_cv1, y_cv1, oscar_coef)
oscar_test_log.append(oscar_test)

####################################OSCAR2###############

OSCAR_X_train = pd.DataFrame(X_train2);
OSCAR_X_test = pd.DataFrame(X_cv2);
OSCAR_y_train = pd.DataFrame(y_train2);
OSCAR_y_test = pd.DataFrame(y_cv2);

OSCAR_X_train.to_csv('OSCAR_X_train.csv', index=False, header=False);
OSCAR_X_test.to_csv('OSCAR_X_test.csv', index=False, header=False);
OSCAR_y_train.to_csv('OSCAR_y_train.csv', index=False, header=False);
OSCAR_y_test.to_csv('OSCAR_y_test.csv', index=False, header=False);

oscar_coef, d_f, tuning= eng.oscar_package_real_data(nargout=3)

oscar_coef=np.array(oscar_coef)
oscar_coef=oscar_coef[:,0];
tuning_log.append(tuning);

oscar_test = lossf(X_cv2, y_cv2, oscar_coef)
oscar_test_log.append(oscar_test)

####################################OSCAR3###############

OSCAR_X_train = pd.DataFrame(X_train3);
OSCAR_X_test = pd.DataFrame(X_cv3);
OSCAR_y_train = pd.DataFrame(y_train3);
OSCAR_y_test = pd.DataFrame(y_cv3);

OSCAR_X_train.to_csv('OSCAR_X_train.csv', index=False, header=False);
OSCAR_X_test.to_csv('OSCAR_X_test.csv', index=False, header=False);
OSCAR_y_train.to_csv('OSCAR_y_train.csv', index=False, header=False);
OSCAR_y_test.to_csv('OSCAR_y_test.csv', index=False, header=False);

oscar_coef, d_f, tuning= eng.oscar_package_real_data(nargout=3)

oscar_coef=np.array(oscar_coef)
oscar_coef=oscar_coef[:,0];
tuning_log.append(tuning);

oscar_test = lossf(X_cv3, y_cv3, oscar_coef)
oscar_test_log.append(oscar_test)

####################################OSCAR4###############

OSCAR_X_train = pd.DataFrame(X_train4);
OSCAR_X_test = pd.DataFrame(X_cv4);
OSCAR_y_train = pd.DataFrame(y_train4);
OSCAR_y_test = pd.DataFrame(y_cv4);

OSCAR_X_train.to_csv('OSCAR_X_train.csv', index=False, header=False);
OSCAR_X_test.to_csv('OSCAR_X_test.csv', index=False, header=False);
OSCAR_y_train.to_csv('OSCAR_y_train.csv', index=False, header=False);
OSCAR_y_test.to_csv('OSCAR_y_test.csv', index=False, header=False);

oscar_coef, d_f, tuning= eng.oscar_package_real_data(nargout=3)

oscar_coef=np.array(oscar_coef)
oscar_coef=oscar_coef[:,0];
tuning_log.append(tuning);

oscar_test = lossf(X_cv1, y_cv1, oscar_coef)
oscar_test_log.append(oscar_test)

####################################OSCAR5###############

OSCAR_X_train = pd.DataFrame(X_train5);
OSCAR_X_test = pd.DataFrame(X_cv5);
OSCAR_y_train = pd.DataFrame(y_train5);
OSCAR_y_test = pd.DataFrame(y_cv5);

OSCAR_X_train.to_csv('OSCAR_X_train.csv', index=False, header=False);
OSCAR_X_test.to_csv('OSCAR_X_test.csv', index=False, header=False);
OSCAR_y_train.to_csv('OSCAR_y_train.csv', index=False, header=False);
OSCAR_y_test.to_csv('OSCAR_y_test.csv', index=False, header=False);

oscar_coef, d_f, tuning= eng.oscar_package_real_data(nargout=3)

oscar_coef=np.array(oscar_coef)
oscar_coef=oscar_coef[:,0];
tuning_log.append(tuning);

oscar_test = lossf(X_cv1, y_cv1, oscar_coef)
oscar_test_log.append(oscar_test)

minindex = np.where(oscar_test_log == np.min(oscar_test_log))

op_tuning = tuning_log[int(minindex[0])]


###############################################################



OSCAR_X_train = pd.DataFrame(X);

OSCAR_y_train = pd.DataFrame(y);

OSCAR_X_train.to_csv('OSCAR_X_train.csv', index=False, header=False);

OSCAR_y_train.to_csv('OSCAR_y_train.csv', index=False, header=False);


oscar_coef, d_f= eng.oscar_package_real_data_finial_model(op_tuning, nargout=2)
oscar_coef = np.array(oscar_coef)
 


######################Elast NET####################################



en_test_log=[]
en_tuning_log=[]


################################1###########################


alphas1 = np.arange(0.01,1,0.05)
alphas2 = np.arange(0.01,1.5,0.05)
elnet = ElasticNet()
coefs = []
errors =[]
optuning=np.array([0,0])

besterror = float("inf")

for a in alphas1:
    for b in alphas2:
        elnet.set_params(alpha=a+b, l1_ratio=a/(a+b))
        elnet.fit(X_train1, y_train1)
        coefs.append(elnet.coef_)
        temperror = lossf(X_cv1, y_cv1,elnet.coef_)
        if temperror < besterror:
            besterror = temperror
            optuning=np.array([a,b])


en_test_log.append(besterror);
en_tuning_log.append(optuning);

#################################2##########################


alphas1 = np.arange(0.01,1,0.05)
alphas2 = np.arange(0.01,1.5,0.05)
elnet = ElasticNet()
coefs = []
errors =[]
optuning=np.array([0,0])

besterror = float("inf")

for a in alphas1:
    for b in alphas2:
        elnet.set_params(alpha=a+b, l1_ratio=a/(a+b))
        elnet.fit(X_train2, y_train2)
        coefs.append(elnet.coef_)
        temperror = lossf(X_cv2, y_cv2,elnet.coef_)
        if temperror < besterror:
            besterror = temperror
            optuning=np.array([a,b])


en_test_log.append(besterror);
en_tuning_log.append(optuning);

##################################3#########################


alphas1 = np.arange(0.01,1,0.05)
alphas2 = np.arange(0.01,1.5,0.05)
elnet = ElasticNet()
coefs = []
errors =[]
optuning=np.array([0,0])

besterror = float("inf")

for a in alphas1:
    for b in alphas2:
        elnet.set_params(alpha=a+b, l1_ratio=a/(a+b))
        elnet.fit(X_train3, y_train3)
        coefs.append(elnet.coef_)
        temperror = lossf(X_cv3, y_cv3,elnet.coef_)
        if temperror < besterror:
            besterror = temperror
            optuning=np.array([a,b])


en_test_log.append(besterror);
en_tuning_log.append(optuning);

##################################4#########################


alphas1 = np.arange(0.01,1,0.05)
alphas2 = np.arange(0.01,1.5,0.05)
elnet = ElasticNet()
coefs = []
errors =[]
optuning=np.array([0,0])

besterror = float("inf")

for a in alphas1:
    for b in alphas2:
        elnet.set_params(alpha=a+b, l1_ratio=a/(a+b))
        elnet.fit(X_train4, y_train4)
        coefs.append(elnet.coef_)
        temperror = lossf(X_cv4, y_cv4,elnet.coef_)
        if temperror < besterror:
            besterror = temperror
            optuning=np.array([a,b])


en_test_log.append(besterror);
en_tuning_log.append(optuning);


##################################5#########################


alphas1 = np.arange(0.01,1,0.05)
alphas2 = np.arange(0.01,1.5,0.05)
elnet = ElasticNet()
coefs = []
errors =[]
optuning=np.array([0,0])

besterror = float("inf")

for a in alphas1:
    for b in alphas2:
        elnet.set_params(alpha=a+b, l1_ratio=a/(a+b))
        elnet.fit(X_train5, y_train5)
        coefs.append(elnet.coef_)
        temperror = lossf(X_cv5, y_cv5,elnet.coef_)
        if temperror < besterror:
            besterror = temperror
            optuning=np.array([a,b])


en_test_log.append(besterror);
en_tuning_log.append(optuning);


minindex = np.where(en_test_log == np.min(en_test_log))

en_op_tuning = en_tuning_log[int(minindex[0])]

a=optuning[0];
b=optuning[1];
elnet.set_params(alpha=a+b, l1_ratio=a/(a+b))
elnet.fit(X, y)

########################################################



lasso_tuning_log=[]
lasso_test_log=[]

###############################1########################




alphas = np.arange(0.01,1.3,0.05)
lasso = Lasso(max_iter=100)
coefs = []
errors =[]

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train1, y_train1)
    coefs.append(lasso.coef_)
    errors.append(lossf(X_cv1, y_cv1, lasso.coef_))

minerror = np.min(errors)
minindex = np.where(errors == minerror)
optuning = alphas[minindex]

lasso_tuning_log.append(optuning)
lasso_test_log.append(minerror)



###################################2#####################


alphas = np.arange(0.01,1.3,0.05)
lasso = Lasso(max_iter=100)
coefs = []
errors =[]

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train2, y_train2)
    coefs.append(lasso.coef_)
    errors.append(lossf(X_cv2, y_cv2, lasso.coef_))

minerror = np.min(errors)
minindex = np.where(errors == minerror)
optuning = alphas[minindex]

lasso_tuning_log.append(optuning)
lasso_test_log.append(minerror)



#####################################3###################
alphas = np.arange(0.01,1.3,0.05)
lasso = Lasso(max_iter=100)
coefs = []
errors =[]

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train3, y_train3)
    coefs.append(lasso.coef_)
    errors.append(lossf(X_cv3, y_cv3, lasso.coef_))

minerror = np.min(errors)
minindex = np.where(errors == minerror)
optuning = alphas[minindex]

lasso_tuning_log.append(optuning)
lasso_test_log.append(minerror)



###################################4#####################
alphas = np.arange(0.01,1.3,0.05)
lasso = Lasso(max_iter=100)
coefs = []
errors =[]

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train4, y_train4)
    coefs.append(lasso.coef_)
    errors.append(lossf(X_cv4, y_cv4, lasso.coef_))

minerror = np.min(errors)
minindex = np.where(errors == minerror)
optuning = alphas[minindex]

lasso_tuning_log.append(optuning)
lasso_test_log.append(minerror)



##################################5######################
alphas = np.arange(0.01,1.3,0.05)
lasso = Lasso(max_iter=100)
coefs = []
errors =[]

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train5, y_train5)
    coefs.append(lasso.coef_)
    errors.append(lossf(X_cv5, y_cv5, lasso.coef_))

minerror = np.min(errors)
minindex = np.where(errors == minerror)
optuning = alphas[minindex]

lasso_tuning_log.append(optuning)
lasso_test_log.append(minerror)



########################################################



minindex = np.where(lasso_test_log == np.min(lasso_test_log))

lasso_op_tuning = lasso_tuning_log[int(minindex[0])]

lasso.set_params(alpha=lasso_op_tuning[0])
lasso.fit(X, y)








































