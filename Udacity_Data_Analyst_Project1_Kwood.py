# In this project, I demonstrate the 'Data Analysis Process' on a public dataset from Kaggle (https://www.kaggle.com/joniarroba/noshowappointments). 
# The dataset contains information about medical appointments in Brazil.

# The project seeks to uncover relationships between the attributes of the patients and the likelihood of them showing up for their scheduled appointment. This is a descriptive analysis. 
# What factors are related to not showing up for appointments? I investigate the following:

# 1) Is there a relationship between gender and no shows?
# 2) Is there a relationship between being on welfare (participation in a public assistance program) and no shows?
# 3) Is there a relationship between gender, welfare program particpation, and no shows?
# 4) is there a relationship between gender, age, and no shows?
 
# In this analysis, the dependent variable is 'no show' and the independent variables are gender (male or female) and welfare program participation (scholarship status).


# Data Wrangling (computer science lingo for data cleaning)
#import packages and assign aliases
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#read data into memory
#set up correct directory... download data from https://www.kaggle.com/joniarroba/noshowappointments or link with API
df=pd.read_csv("/Users/user/Downloads/noshowappointments-kagglev2-may-2016.csv")
# I view the raw data with the 'head()' command.
df.head()

df.shape
# The dataset has 110,527 observations and 14 variables.
df.info()
# None of the observations have any missing values for the 14 variables. This is evidenced by the fact the the number of non-null observations for each variable (column) is equal to the total number of observations

#Change the variables for the 'Scheduled Day' and 'Appointment Day' to datetime.
pd.to_datetime(df['ScheduledDay']);
pd.to_datetime(df['AppointmentDay']);

#Rename mis-spelled columns and lower case all columns
df=df.rename(columns={'Hipertension': 'Hypertension', 'Handcap' : 'Handicap', 'No-show': "No_show"})
df.columns = map(str.lower, df.columns)
df.info()

#Rename the 'No-show' column/variable with an underscore, 'No_show', so that I can use in inline with certain commands. 
# Specfically, I change the key dependent variable 'no_show' from string to a binary integer. This allows me to take the mean to find the percent of the respective group that is a 'no show' rather than having to create the proportion of the group (count of group that 'no show' relative to total)
df['no_show'].replace(('Yes', 'No'), (1, 0), inplace=True)
#Change no show and gender to binary variables
df['gender'].replace(('M', 'F'), (1, 0), inplace=True)
df=df.rename(columns={'gender': "male"})

df.nunique()
#There are 62,299 unique patients in the dataset. Therefore, there are some observations represent multiple visits from the same patient since the dataset has 110,527 total observations.
# check for duplicates in the data
sum(df.duplicated())
# There are no duplicates in the dataset

df.hist(figsize=(15,15));
# The histograms suggest that:
# 1) most patients are female
# 2) most patients are in good health (with respect to hypertension, alcoholism, diabetes, and general handicaps)
# 3) a minority of patients receive a sms text message

#Exploratory Data Analysis (I answer a few basic questions with the data)

df.no_show.mean()
# On average, about 1 in every 5 appointments results in a 'no show'


# I create a user written function to divide column values and tabulate them. I use it below to tabulate the appointments 
# by neighborhood and appointment day.
def tab(col_name):
    #Take a column, and separate with '|'
    tab = df[col_name].str.cat(sep = '|')    
    #Make it into a series and store the values separately
    tab = pd.Series(tab.split('|'))
    #Sort in descending order so that the first element is the most frequently-occurring element.
    frequency = tab.value_counts(ascending = False)
    return frequency

hood = tab('neighbourhood')
hood.head(10)
# There are 81 neighbourhoods that appointments in this dataset are made from. We can see that many of the appointments come from 
# the top 10 neighborhoods. Below I examine the bottom 10 neighborhoods.
hood.tail(10)
# Relative to the top 10 neighborhoods, barely any appointments are made for the bottom 10. 

day=tab('appointmentday')
day
# This dataset has 27 different appointment days. All days have at least 3000 appointments besides 2016-05-14 where only 39 
# appointments were recorded. This may be an error in the dataset. 

# I check if there is a difference between the genders for no shows.
# Use query to select each group and get its mean 
male = df.query('male ==1')
female = df.query('male ==0')

mean_noshow_male = male['no_show'].mean()
mean_noshow_female = female['no_show'].mean()

count_noshow_male=male['no_show'].count()
count_noshow_female=female['no_show'].count()

print(str(format(mean_noshow_male*100,"0.2f")) 
      + ' percent of Males do not show up for their appointments while ' 
      + str(format(mean_noshow_female*100,"0.2f")) 
      + ' percent of females do not show')

df.male.mean()
# There is a very small difference between the percent of males and females missing appointments in the total sample. However, 
# only 35 percent of the all appointments are for male patients so a greater proportion of the absolute amount of no shows are for females.


# VISUAL REPRESENTATIONS OF THE DATA

# I create a bar chart with of the likelihood of a No Show by gender
locations = [1, 2]
heights = [mean_noshow_male, mean_noshow_female]
labels = ['Male', 'Female']
plt.bar(locations, heights, tick_label=labels)
plt.title('Likelihood of No Show Appointments by Gender')
plt.xlabel('Gender')
plt.ylabel('Likelihood of No Show');
#The likelihood of missing an appointment appears to be almost the same for the genders.


# I create a bar chart on counts of missed appointments by gender
locations = [1, 2]
heights = [count_noshow_male, count_noshow_female]
labels = ['Male', 'Female']
plt.bar(locations, heights, tick_label=labels)
plt.title('Count of No Show Appointments by Gender')
plt.xlabel('Gender')
plt.ylabel('Count of No Shows');
#There are many money females missing appointments than males, in an absolute sense. However, as seen in the previous figure, both genders are almost just as likely to miss an appointment.


# I check if scholarship (welfare) recipients less likely to show up for an appointment?
# Query to select each group by scholarship status and get its mean
scholarship = df.query('scholarship ==1')
no_scholarship = df.query('scholarship ==0')

mean_noshow_sch = scholarship['no_show'].mean()
mean_noshow_nosch = no_scholarship['no_show'].mean()

count_noshow_sch = scholarship['no_show'].count()
count_noshow_nosch = no_scholarship['no_show'].count()

print(str(format(mean_noshow_sch*100,"0.2f")) 
      + ' percent of scholarship recipients do not show up for their appointments while ' 
      + str(format(mean_noshow_nosch*100,"0.2f")) 
      + ' percent of non-scholarship patients do not show. '
     'Scholarships are awarded to poor families. There may be transportation issues ' 
      + 'leading to more no shows, or there may be issues with reliability of the '
      + 'patients due to moral hazard (no having to bear the cost themselves).')
# 23.74 percent of scholarship recipients do not show up for their appointments while 19.81 percent of non-scholarship patients do not show. Scholarships are awarded to poor families. There may be transportation issues leading to more no shows, or there may be issues with reliability of the patients due to moral hazard (no having to bear the cost themselves).

print('However, only ' 
      + str(format(df['scholarship'].mean()*100,"0.2f")) 
      + ' percent of the total patient population are on scholarship' 
      + ' so they are not driving the majority of no shows.')
# However, only 9.83 percent of the total patient population are on scholarship so they are not driving the majority of no shows.

# I create a bar chart on average likelihood of missed appointment by scholarship status
locations = [1, 2]
heights = [mean_noshow_sch, mean_noshow_nosch]
labels = ['Scholarhsip', 'No Scholarship']
plt.bar(locations, heights, tick_label=labels)
plt.title('Likelihood of No Show Appointments by Scholarship Status')
plt.xlabel('Scholarship Status')
plt.ylabel('Percent No Show');
#Scholarship recipients are more likely to not show up for their appointments relative to non-scholarship recipients.

# I create a bar chart on counts of no shows by scholarship status
locations = [1, 2]
heights = [count_noshow_sch, count_noshow_nosch]
labels = ['Scholarhsip', 'No Scholarship']
plt.bar(locations, heights, tick_label=labels)
plt.title('Count of No Show Appointments by Scholarship Status')
plt.xlabel('Scholarship Status')
plt.ylabel('Count of No Show');
# There are many more no show appointments among the no scholarship group. This is because scholarship recipients account for only 10% of the total appointments in the dataset.


# Is there a relationship between gender, welfare particpation, and no shows?
# Query to select each group by scholarship status and gender to creat four subset dataframes
df_male_sch=df.query('scholarship==1 & male==1')
df_female_sch=df.query('scholarship==1 & male==0')

df_male_nosch=df.query('scholarship==0 & male==1')
df_female_nosch=df.query('scholarship==0 & male==0')

# Compute the means of each respective df
mean_noshow_male_sch=df_male_sch.no_show.mean()

mean_noshow_female_sch=df_female_sch.no_show.mean()

mean_noshow_male_nosch=df_male_nosch.no_show.mean()

mean_noshow_female_nosch=df_male_nosch.no_show.mean()

# Create a bar chart showing the likelihood of no show by gender and scholarship status 
locations = [1, 2, 3, 4]
heights = [mean_noshow_male_sch, mean_noshow_female_sch, 
           mean_noshow_male_nosch, mean_noshow_female_nosch]
labels = ['Male Scholarship', 'Female Scholarship', 'Male No Scholarship', 'Female No Scholarship']
plt.bar(locations, heights, tick_label=labels)
plt.title('Likelihood of No Show Appointments by Scholarship Status and Gender')
plt.xlabel('Gender and Scholarship Status')
plt.ylabel('Likelihood of No Show')
fig = plt.gcf()
fig.set_size_inches(15, 10)
fig;
#Females on Scholarship are most likely to miss their appointments relative to all others. 
# Male scholarship recipients are more likely to miss their appointment than either gender 
# with no scholarship. There is not a visually distinguishable difference between the genders 
# and the likelihood of missing an appointment for the no scholarship group.


#create multi-dimension box plot figure with gender, age, and no show.
sns.catplot(x="male", y="age", col="no_show", data=df, height=6, kind="box" )
# Lastly, I add additional dimensions to the analysis by considering if there is a relationship between age, gender, and the likelihood of not showing up for an appointment. The figure above shows (in blue boxes) that females tend to be older regardless of not showing up for the appointment. However, younger females are more likely not to show. This trend of younger individuals being less likely to show up is consistent with both genders.

# # Conclusion and Limitation

# This project explored the Kaggle Brazilian medical appointment dataset and attempted to uncover relationships between gender, welfare program participation, and the likelihood of missing an appointment. 
# The data was loaded into python and standard data procedures were performed such as: 
# 1) searching for missing values
# 2) searching for duplicated rows of data
# 3) reformatting variables and
# 4) renaming columns
# The exploratory data analysis focused on the gender, scholarship status (welfare receipt), and the likelihood of missing an appointment.
# Only about 35 percent of appointments were for males, and only 10 percent of appointments were for scholarship recipients. However, the most likely type of individuals to miss appointments were female scholarship recipients. 
# A limitation of this analysis was that it was only descriptive. Some of the mean statistics in this analysis appear to be very close numbers. This analysis could be improved by conducting 't-testing' to formally find out if there exists a statistically significant different in the mean among these groups.
