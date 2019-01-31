"""
Steps:
1. Generate universe variables
2. Generate fuzzy membership functions
3. Find the activation of membership functions at inputs
4. Based on Rules apply OR/AND operator
5. Getting Output of each membership function
6. Aggregation - to combine all outputs of membership function 
7. Defuzzification - to get crisp value from aggregated fuzzy function

Problem:
Inputs: 1. Service Quality(bad(0-4),mod(4-7),great(7-10)) - [0,10]
        2. Food Quality(terrible(0-5),delicious(5-10)) - [0,10]
Output: 1. Tip Percentage(lo(0-10),med(10-18),hi(18-25)) -[0,25]
Rules: 1. IF the service was good or the food quality was good, THEN the tip will be high.
       2. IF the service was average, THEN the tip will be medium.
       3. IF the service was poor and the food quality was poor THEN the tip will be low.

Implication     :
Aggregation     : 
AND             : min
OR              : max
Defuzzification : Centroid
!There are 504,000 combinations of parameter sets that i need to test for the tipping problem which is Impractical
""" 
import matplotlib.pyplot as plt
import numpy as np 
import skfuzzy as fuzz
from mfArea import mfAreaOverlap as mfA

# Declaring Universe input and output variables of the fuzzy system
service = np.arange(0,11,1)
food = np.arange(0,11,1)
tip = np.arange(0,26,1)

#***********Defining membership functions for input and outputs*******************
#service - 27
i=1
Area = np.zeros((60,2))
allServiceMf=list()
z=0
while(i<4):
    serviceBad = fuzz.trimf(service,[1,i,4]) # 1-4 range; <1 0% and >=4 0%
    i+=1
    j=3
    while(j<7):
        serviceMod = fuzz.trimf(service, [3,j,7]) # 4-7 range; <4 0% and >=7 0%
        j+=1
        k=6
        while(k<=10):
            serviceGreat = fuzz.trimf(service, [6,k,10]) # 7-10 range; <7 0% and >10 0%
            concat = serviceBad,serviceMod,serviceGreat
            Area[z][0] = mfA.trimfArea(service,[1,i,4],[3,j,7])
            Area[z][1] = mfA.trimfArea(service,[3,j,7],[6,k,10])
            allServiceMf.append(concat)
            z+=1
            k+=1
                     
#food - 15 
l=2
allFoodMf=list()
while(l<5):
    foodTerr = fuzz.trimf(food, [2,l,5]) # 2-5 range; <2 0% and >=5 0%
    m=5
    l+=1
    while(m<=10):
        foodDel = fuzz.trimf(food, [5,m,10]) # 5-10 range; <5 0% and >10 0%
        m+=1
        concat = foodTerr,foodDel
        allFoodMf.append(concat)

#tip - 448
n=2
allTipMf=list()
while(n<10):
    tipLo = fuzz.trimf(tip, [2,n,10]) # 2-10 range; <2 0% and >=10 0%
    o=10
    n+=1
    while(o<17):
        tipMed = fuzz.trimf(tip, [10,o,17]) # 10-17 range; <10 0% and >=17 0%
        o+=1
        p=17
        while(p<=25):
            tipHi = fuzz.trimf(tip, [17,p,25]) # 17-25 range; <17 0% and >25 0%
            p+=1
            concat = tipLo,tipMed,tipHi
            allTipMf.append(concat)

# Plotting membership functions
fig,  (ax0,ax1,ax2) = plt.subplots(nrows=1, figsize=(5, 6))

ax0.plot(food, allFoodMf[7][0], 'b', linewidth=1.5, label='Terrible')
ax0.plot(food, allFoodMf[7][1], 'g', linewidth=1.5, label='Delicious')
ax0.set_title('Food quality')
ax0.legend()

ax1.plot(service, allServiceMf[59][0], 'b', linewidth=1.5, label='Bad')
ax1.plot(service, allServiceMf[59][1], 'g', linewidth=1.5, label='Moderate')
ax1.plot(service, allServiceMf[59][2], 'r', linewidth=1.5, label='Great')
ax1.set_title('Service quality')
ax1.legend()

ax2.plot(tip, allTipMf[70][0], 'b', linewidth=1.5, label='Low')
ax2.plot(tip, allTipMf[70][1], 'g', linewidth=1.5, label='Medium')
ax2.plot(tip, allTipMf[70][2], 'r', linewidth=1.5, label='High')
ax2.set_title('Tip Percentage')
ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
plt.tight_layout()

plt.show()

#****************Applying Rules and Implication************************************

#Getting membership value for the Universe value 
fIn=5
sIn=9
foodLevTerr = fuzz.interp_membership(food, allFoodMf[5][0], fIn)
foodLevDel = fuzz.interp_membership(food, allFoodMf[5][1], fIn)
serviceLevBad = fuzz.interp_membership(service, allServiceMf[4][0], sIn)
servLevMod = fuzz.interp_membership(service, allServiceMf[4][1], sIn)
serviceLevGreat = fuzz.interp_membership(service, allServiceMf[4][2], sIn)

# Rule1 + Implication(min)
rule1 = np.fmax(foodLevTerr, serviceLevBad)
tipImpLo = np.fmin(rule1, tipLo)  # removed entirely to 0

# Rule2 + Implication(min)
tipImpMed = np.fmin(servLevMod, tipMed)

# Rule3 + Implication(min)
rule3 = np.fmax(foodLevDel, serviceLevGreat)
tipImpHi = np.fmin(rule3, tipHi)
tip0 = np.zeros_like(tip)

# Plotting Output membership activity
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(tip, tip0, tipImpLo, facecolor='b', alpha=0.7)
ax0.plot(tip, allTipMf[70][0], 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(tip, tip0, tipImpMed, facecolor='g', alpha=0.7)
ax0.plot(tip, allTipMf[70][1], 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(tip, tip0, tipImpHi, facecolor='r', alpha=0.7)
ax0.plot(tip, allTipMf[70][2], 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

#********************Applying Aggregation(max)*******************************

agg = np.fmax(tipImpLo,np.fmax(tipImpMed, tipImpHi))

#*******************Applying Centroid Defuzzification************************

tipCrisp = fuzz.defuzz(tip, agg, 'centroid')
print("\nThe final crisp value is: ")
print(tipCrisp)
tipAgg = fuzz.interp_membership(tip, agg, tip)  # for plot

# Plotting Aggregated membership function with defuzzified crisp value
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(tip, allTipMf[70][0], 'b', linewidth=0.5, linestyle='--', )
ax0.plot(tip, allTipMf[70][1], 'g', linewidth=0.5, linestyle='--')
ax0.plot(tip, allTipMf[70][2], 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(tip, tip0, agg, facecolor='Orange', alpha=0.7)
ax0.plot([tipCrisp, tipCrisp], [0, tipAgg], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated Membership and Defuzzified Crisp Value')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
# Display all plots
plt.plot(service,fuzz.gaussmf(service,5,2), 'b', linewidth =1)
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),
          ncol=1, fancybox=True, shadow=True);
plt.show()
