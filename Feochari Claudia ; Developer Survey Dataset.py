#!/usr/bin/env python
# coding: utf-8

# # Claudia Feochari , 40000060

# # Preliminary Report

# # Women in Tech HackerRank Developer Survey

# ## Dataset Retrieval

# # Questions to Answer

# ### In which age range are most female participants part of ?
# ### What are the percentage of women versus men involved in tech in Canada versus in India?
# ### Has the female participant been in school? If so, what kinds of schooling has she done?
# ### When is the average age that women in general first start to learn how to code?
# ### What are the most common types of jobs that women who took the survey have?
# ### What are the main interest areas of women in? Eg: do most of them prefer AI technologies or IOT for instance)
# ### What are the languages that most women participants do not like?
# ### Which resources did the women use to learn how to code? 
# ### What differences are there between the responses from the professional women versus the female student responses? For instance, what kinds of IDE’s do the female professionals use versus the female students? 
# ### What do young women in the 18-24 years age do for a living?
# ### What percentage of female participants answered the coding question wrong? Which age category do these female participants belong to mostly? What do they do for a living?
# ### Are women more interested in back-end technologies or front-end ones? 
# ### What are the employment levels of women compared to men? (Junior versus Senior developer)
# 

# ## Here, we are creating a directory where all of the dataset files will be stored

# In[2]:


### Importing a Python 3 version of "urlib"
import urllib.request
import os
import gzip

### If not existing, create the data folder locally , else do nothing
# Creating the datasets folder if it doesn't exist

if not os.path.exists('./datasets'):
    os.makedirs('./datasets')


# ## Collecting the datasets uploaded to GitHub

# ## Retrieval Steps to get the data used for the data analysis:
# ### Rather than manually downloading the data to a local folder, it can be retrieved using the links provided. To retrieve the data sets, a script was written using the urllib.request library in python. The original dataset originates from kaggle.com but retrieving the code using the link shown below returned an error when trying to retrieve the data that will be used for analysis. 
# 
# ### https://www.kaggle.com/hackerrank/developer-survey-2018/downloads/developer-survey-2018.zip/3  
# 
# ### Since the direct retrieval was not working from this website, I decided to take an alternate route when gathering the data. I decided that it would be best to upload the data to a GitHub repository that I had created on my account. One might assume that this might be problematic as the data would not be the updated data but this is not the case as the dataset comes from a survey done at a point in time (late 2016) , so the files would not be updated anymore. The source shown below is the folder in which each of the dataset csv files are stored in. 
# 
# 
# ### https://github.com/compgirl123/COMP-499-DataSets/tree/master/developer-survey-2018%20(9)
# 

# ## Collection of First DataSet HackerRank Survey 2018

# # Data Wrangling

# ## Discovery Step (Data Wrangling)
# ### Seeing what data is available

# In[3]:


# Obtaining the dataset using the url that hosts it
import os
import urllib.request
country_code_url = 'https://raw.githubusercontent.com/compgirl123/COMP-499-DataSets/master/developer-survey-2018%20(9)/Country-Code-Mapping.csv'
if not os.path.exists('./datasets/country_code_mapping_hackerrank_dev_survey.csv'):     # avoid downloading if the file exists
    response = urllib.request.urlretrieve(country_code_url, './datasets/country_code_mapping_hackerrank_dev_survey.csv')


# In[4]:


# Obtaining the dataset using the url that hosts it
import os
import urllib.request
codebook_url = 'https://raw.githubusercontent.com/compgirl123/COMP-499-DataSets/master/developer-survey-2018%20(9)/HackerRank-Developer-Survey-2018-Codebook.csv'
if not os.path.exists('./datasets/codebook_hackerrank_dev_survey.csv'):     # avoid downloading if the file exists
    response = urllib.request.urlretrieve(codebook_url, './datasets/codebook_hackerrank_dev_survey.csv')


# In[5]:


# Obtaining the dataset using the url that hosts it
import os
import urllib.request
numeric_mapping_url = 'https://raw.githubusercontent.com/compgirl123/COMP-499-DataSets/master/developer-survey-2018%20(9)/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv'
if not os.path.exists('./datasets/numeric_mapping_hackerrank_dev_survey.csv'):     # avoid downloading if the file exists
    response = urllib.request.urlretrieve(numeric_mapping_url, './datasets/numeric_mapping_hackerrank_dev_survey.csv')


# In[6]:


# Obtaining the dataset using the url that hosts it
import os
import urllib.request
numeric_hackerrank_url = 'https://raw.githubusercontent.com/compgirl123/COMP-499-DataSets/master/developer-survey-2018%20(9)/HackerRank-Developer-Survey-2018-Numeric.csv'
if not os.path.exists('./datasets/numeric_hackerrank_dev_survey.csv'):     # avoid downloading if the file exists
    response = urllib.request.urlretrieve(numeric_hackerrank_url, './datasets/numeric_hackerrank_dev_survey.csv')


# In[2165]:


# Obtaining the dataset using the url that hosts it
import os
import urllib.request
values_url = 'https://raw.githubusercontent.com/compgirl123/COMP-499-DataSets/master/developer-survey-2018%20(9)/HackerRank-Developer-Survey-2018-Values.csv'
if not os.path.exists('./datasets/values_hackerrank_dev_survey.csv'):     # avoid downloading if the file exists
    response = urllib.request.urlretrieve(values_url, './datasets/values_hackerrank_dev_survey.csv')


# ## Collection of Second DataSet StackOverflow Survey 2018

# ### Obtaining the dataset by downloading it directly as uploading .csv to GitHub did not work as it was too big.
# ### Stackoverflow

# ### https://www.kaggle.com/stackoverflow/stack-overflow-2018-developer-survey/downloads/stack-overflow-2018-developer-survey.zip/2
# 

# ## Loading of each Data set as a DataFrame

# ### HackRank Country Code Data

# In[2162]:


import pandas as pd
country_code_data = pd.read_csv('./datasets/country_code_mapping_hackerrank_dev_survey.csv')


# In[2163]:


country_code_data.head()


# ### HackRank Codebook Data

# In[1150]:


import pandas as pd
codebook_data = pd.read_csv('./datasets/codebook_hackerrank_dev_survey.csv')


# In[1151]:


codebook_data.head()


# ### HackRank Numeric Data Mapping

# In[1152]:


import pandas as pd
numeric_mapping_data = pd.read_csv('./datasets/numeric_mapping_hackerrank_dev_survey.csv')


# In[1277]:


numeric_mapping_data.head()


# ### HackRank Numeric Data

# In[1154]:


import pandas as pd
numeric_data = pd.read_csv('./datasets/numeric_hackerrank_dev_survey.csv')


# In[1155]:


numeric_data.head()


# ### HackRank Values Data

# In[1745]:


import pandas as pd
values_data = pd.read_csv('./datasets/values_hackerrank_dev_survey.csv')


# In[1746]:


values_data.head()


# ### StackOverFlow Survey Data 2018

# In[1158]:


total_data_stackoverflow = pd.read_csv('./datasets/stack-overflow-2018-developer-survey/survey_results_public.csv')


# In[1159]:


total_data_stackoverflow.head()


# ## Creating Subsets for Our Data

# ### Filtering by females in HackerRank Dataset

# ### Data cleaning replacing #NULL! with NaN and removing NaN for q1AgeBeginCoding

# In[1978]:


female_values_data = values_data[values_data['q3Gender'] == "Female"]
female_values_data.replace("#NULL!",np.nan)
female_values_data = female_values_data.dropna(subset=['q1AgeBeginCoding'])


# In[1979]:


female_values_data


# ### Filtering by males in HackerRank Dataset

# ### Data cleaning replacing #NULL! with NaN and removing NaN for q1AgeBeginCoding

# In[1980]:


male_values_data = values_data[values_data['q3Gender'] == "Male"]
male_values_data.replace("#NULL!",np.nan)
male_values_data = male_values_data.dropna(subset=['q1AgeBeginCoding'])


# In[1981]:


male_values_data.head()


# ### Filtering by females in StackOverflow Dataset

# In[1164]:


women_stack_overflow = total_data_stackoverflow.loc[total_data_stackoverflow['Gender'] == "Female"]


# In[1165]:


women_stack_overflow.head()


# ### Filtering by males in StackOverflow Dataset

# In[1166]:


men_stack_overflow = total_data_stackoverflow.loc[total_data_stackoverflow['Gender'] == "Male"]


# In[1167]:


men_stack_overflow.head()


# # Descriptive Analysis

# ### Total Number of Participants

# In[1294]:


# Size of the dataset, number or rows
values_data.shape[0]


# ### The Amount of Women Participants in the Survey

# In[1295]:


# Amount of Women Participants that participated in the Survey present in the values_data DataFrame, the DataFrame
# containing all of the RAW information about the participants.
values_female_participants = values_data.loc[values_data['q3Gender'] == "Female"]
values_female_participants.shape[0]


# ### The Amount of Men Participants in the Survey

# In[2190]:


# Amount of Women Participants that participated in the Survey present in the values_data DataFrame, the DataFrame
# containing all of the RAW information about the participants.
values_male_participants = values_data.loc[values_data['q3Gender'] == "Male"]
values_male_participants.shape[0]


# ### View sample Top 5 Entries of  Female Data

# In[15]:


values_female_participants.head()


# ## Five Number Summary of a column in HackerRank Numeric DataFrame

# ### Gathering the numeric data, the DataFrame containing numbers instead of labels and that Maps to another DataFrame. FIltering the DataFrame According to the Age of the participant

# In[1168]:


filtered_numeric_data = numeric_data.dropna(axis=0, subset=['q2Age'])
filtered_numeric_data.head()


# ### Adding the Gender Filter, Filtering by "2" which represents Females

# In[1169]:


female_participants_numeric_data = filtered_numeric_data.loc[filtered_numeric_data['q3Gender'] == "2"]
female_participants_numeric_data.head()


# ### Replacing '#NULL!' in the age column with NaN

# In[1170]:


import numpy as np

female_participants_numeric_data['q2Age'] = female_participants_numeric_data['q2Age'].replace('#NULL!',np.nan)
filtered_female_participants_numeric_data = female_participants_numeric_data.dropna(axis=0, subset=['q2Age'])


# ### Converting from String to float32

# In[1171]:


filtered_female_participants_numeric_data.loc["q2Age",:]=pd.to_numeric(filtered_female_participants_numeric_data.q2Age, errors='coerce')
filtered_female_participants_numeric_data.loc["q2Age",:]= filtered_female_participants_numeric_data.q2Age.astype('float32')


# ### Showing the 5 number summary for the Age Category Column

# In[2169]:


filtered_female_participants_numeric_data.head()


# ### Describing the column q2Age 

# In[1801]:


filtered_female_participants_numeric_data['q2Age'].dropna().astype(int).describe()


# #### Five Number Summary
# #### Maximum : 9.0
# #### Third Quartile Q3(75%) : 4.0
# #### Median: 3.0
# #### Third Quartile Q1 (25%) : 3.0
# #### Minimum: 1.0

# ### Describing all of the columns present in the HackerRank Numeric Survey DataFrame

# In[1174]:


numeric_data.describe(include='all')


# In[1175]:


total_data_stackoverflow.describe(include='all')


# In[1176]:


total_data_stackoverflow.describe(include='all')['ConvertedSalary']


# ### Describing all of the columns present in the StackOverFlow Survey DataFrame

# ### Listing the first 10 columns present in the HackerRank DataFrame

# In[1177]:


list_of_columns = values_data.columns.values.tolist()
list_of_columns[0:10]


# ### Data Types for first 10 columns in values_data DataFrame

# In[1178]:


a = 0
for x in values_data:
    if (numeric_data[x].dtype == 'float64' or numeric_data[x].dtype == 'int64') and a<10:
        data = numeric_data[x]
        output = "name:{}, dtype:{}".format(data.name,data.dtype)
        print(output) 
        a+=1


# ### Showing the subset of Data from the CSV that is useful for the Age column

# In[1179]:


df_age = numeric_mapping_data[numeric_mapping_data['Data Field'] == "q1AgeBeginCoding"]
df_age


# ### Displaying a Histogram with Information on the Age Ranges of the Participants in the Survey

# ### Labels
# ### 1: Under 12 years old
# ### 2: 12 - 18 years old
# ### 3: 18 - 24 years old
# ### 4: 25 - 34 years old
# ### 5: 35 - 44 years old
# ### 6: 45 - 54 years old
# ### 7: 55 - 64 years old
# ### 8: 65 - 74 years old
# ### 9: 75 years or older

# ### Converting Ages to Integers

# In[1217]:


string_array = filtered_female_participants_numeric_data['q2Age'].dropna(how='any').values
int_array = string_array.astype('int')


# In[1235]:


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

num_bins = 9
n, bins, patches = plt.hist(int_array, num_bins, facecolor='blue', alpha=0.5)
plt.show()


# ### Displaying a BoxPlot with Information on the Age Ranges of the Participants in the Survey

# In[1243]:


import matplotlib.pyplot as plt
 
box_plot_data=int_array
plt.boxplot(box_plot_data)
plt.xticks([1], ['Ages'])
plt.show()


# In[2170]:


import numpy as np

female_participants_numeric_data['q2Age'] = female_participants_numeric_data['q2Age'].replace('#NULL!',np.nan)
filtered_female_participants_numeric_data = female_participants_numeric_data.dropna(axis=0, subset=['q2Age'])


# ### Data Mapping of the Data From one DataFrame to Another

# In[2171]:


filtered_female_participants_numeric_data['q2Age'] = filtered_female_participants_numeric_data['q2Age'].fillna(0)
filtered_female_participants_numeric_data['q2Age'] = pd.to_numeric(filtered_female_participants_numeric_data['q2Age'])
df_age.merge(filtered_female_participants_numeric_data, left_on='Value', right_on='q2Age')


# In[57]:


values_female_participants.mean()


# In[1134]:


values_female_participants.isnull().sum().head()


# ### Using describe() on the female_participants DataFrame to describe the age they started coding

# In[17]:


values_female_participants['q1AgeBeginCoding'].describe()


# ### Using describe() on the female_participants DataFrame to describe the Degree Focus

# In[18]:


values_female_participants['q5DegreeFocus'].describe()


# # Data Wrangling

# ## Data Profiling and Cleaning

# ## Data Profiling and Structuring : Calculating some basic statistics 

# In[1254]:


country_code_data.head()


# ### Finding out the amount of entries in each DataFrame

# In[1261]:


array_of_dataframes = [country_code_data,codebook_data,numeric_mapping_data,numeric_data,values_data,
                       total_data_stackoverflow,female_values_data,male_values_data,women_stack_overflow,
                      men_stack_overflow]
string_array_of_dataframe = ['country_code_data','codebook_data','numeric_mapping_data','numeric_data','values_data',
                       'total_data_stackoverflow','female_values_data','male_values_data','women_stack_overflow',
                      'men_stack_overflow']
for x in range(0,len(array_of_dataframes)):
    print("Dataframe:"+string_array_of_dataframe[x]+":{}".format(array_of_dataframes[x].shape[0]))


# ### Checking For Duplicates 

# In[1286]:


print('Number of duplicates in ratings_data: {}'.format(sum(country_code_data.duplicated(subset=['Value'], keep=False))))
print('Number of duplicates in ratings_data: {}'.format(sum(codebook_data.duplicated(subset=['Data Field'], keep=False))))

print('Number of duplicates in ratings_data: {}'.format(sum(numeric_data.duplicated(subset=['RespondentID'], keep=False))))
print('Number of duplicates in ratings_data: {}'.format(sum(values_data.duplicated(subset=['RespondentID'], keep=False))))
print('Number of duplicates in ratings_data: {}'.format(sum(female_values_data.duplicated(subset=['RespondentID'], keep=False))))
print('Number of duplicates in ratings_data: {}'.format(sum(male_values_data.duplicated(subset=['RespondentID'], keep=False))))
      
print('Number of duplicates in ratings_data: {}'.format(sum(total_data_stackoverflow.duplicated(subset=['Respondent'], keep=False))))
print('Number of duplicates in ratings_data: {}'.format(sum(women_stack_overflow.duplicated(subset=['Respondent'], keep=False))))
print('Number of duplicates in ratings_data: {}'.format(sum(men_stack_overflow.duplicated(subset=['Respondent'], keep=False))))


# ### Dealing with duplicates (Cleaning)

# ### Only keeping first occurrence of a duplicated entry and the rest is removed

# In[1290]:


country_code_data = country_code_data.drop_duplicates(subset=['Value'], keep='first').copy()
codebook_data=codebook_data.drop_duplicates(subset=['Data Field'], keep='first').copy()

numeric_data=numeric_data.drop_duplicates(subset=['RespondentID'], keep='first').copy()
values_data=values_data.drop_duplicates(subset=['RespondentID'], keep='first').copy()
female_values_data=female_values_data.drop_duplicates(subset=['RespondentID'], keep='first').copy()
male_values_data=male_values_data.drop_duplicates(subset=['RespondentID'], keep='first').copy()

total_data_stackoverflow=total_data_stackoverflow.drop_duplicates(subset=['Respondent'], keep='first').copy()
women_stack_overflow=women_stack_overflow.drop_duplicates(subset=['Respondent'], keep='first').copy()
men_stack_overflow=men_stack_overflow.drop_duplicates(subset=['Respondent'], keep='first').copy()


# ### Normalizing the text (cleaning) By Finding the percentages of each column by removing NaN's in order to 
# ### Use this data and plot the graph for women's preferred languages

# In[1783]:


female_values_data['q22LangProfC'].replace("#NULL!", "NaN")

values_df_columns_list = list(female_values_data.columns.values)

for z in values_df_columns_list:
    if "q22LangProf" in z:
        female_values_data[z] = female_values_data[z].replace("#NULL!", np.nan)
        #print(values_data[z])

agnostic = female_values_data.q22LangProfAgnostic.dropna().shape[0]/female_values_data.shape[0] * 100
profc = female_values_data.q22LangProfC.dropna().shape[0]/female_values_data.shape[0] * 100
cplusplus = female_values_data.q22LangProfCPlusPlus.dropna().shape[0]/female_values_data.shape[0] * 100
langprofjava = female_values_data.q22LangProfJava.dropna().shape[0]/female_values_data.shape[0] * 100
langprofpython = female_values_data.q22LangProfPython.dropna().shape[0]/female_values_data.shape[0] * 100
ruby = female_values_data.q22LangProfRuby.dropna().shape[0]/female_values_data.shape[0] * 100
javascript = female_values_data.q22LangProfJavascript.dropna().shape[0]/female_values_data.shape[0] * 100
CSharp = female_values_data.q22LangProfCSharp.dropna().shape[0]/female_values_data.shape[0] * 100
langProfGo = female_values_data.q22LangProfGo.dropna().shape[0]/female_values_data.shape[0] * 100
langProfScala = female_values_data.q22LangProfScala.dropna().shape[0]/female_values_data.shape[0] * 100
langProfPerl = female_values_data.q22LangProfPerl.dropna().shape[0]/female_values_data.shape[0] * 100
langProfSwift = female_values_data.q22LangProfSwift.dropna().shape[0]/female_values_data.shape[0] * 100
langProfPascal = female_values_data.q22LangProfPascal.dropna().shape[0]/female_values_data.shape[0] * 100
langProfClojure = female_values_data.q22LangProfClojure.dropna().shape[0]/female_values_data.shape[0] * 100
langProfPHP = female_values_data.q22LangProfPHP.dropna().shape[0]/female_values_data.shape[0] * 100
langProfHaskell= female_values_data.q22LangProfHaskell.dropna().shape[0]/female_values_data.shape[0] * 100
langProfLua = female_values_data.q22LangProfLua.dropna().shape[0]/female_values_data.shape[0] * 100
langProfR= female_values_data.q22LangProfR.dropna().shape[0]/female_values_data.shape[0] * 100

array_of_percentages = [agnostic,profc,cplusplus,langprofjava,langprofpython,ruby,javascript,CSharp,
                       langProfGo,langProfScala,langProfPerl,langProfSwift,langProfPascal
                        ,langProfClojure,langProfPHP,langProfHaskell,langProfLua,langProfR]
array_of_percentages
array_of_percentages


# In[1793]:


import matplotlib.pyplot as plt
import numpy as np
label = ['agnostic','C','C++','java','python','ruby','javascript','CSharp',
                       'Go','Scala','Perl','Swift','Pascal','Clojure','PHP'
         ,'Haskell','Lua','R']


def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, array_of_percentages,color='m')
    plt.xlabel('Name of Language', fontsize=15)
    plt.ylabel('Percent of Proficiency (%)', fontsize=15)
    plt.xticks(index, label, fontsize=10, rotation=40)
    plt.title('Proficiency of Languages for Women')
    plt.show()
   
plot_bar_x()


# ### Creating samples that will be used in the Exploratory Data Analysis Step

# In[1291]:


men_stack_overflow = total_data_stackoverflow.loc[total_data_stackoverflow['Gender'] == "Male"]
men_currently_usd = men_stack_overflow.dropna(axis=0, subset=['CurrencySymbol','ConvertedSalary','ConvertedSalary'])
men_currently_usd = men_currently_usd.loc[(men_currently_usd['CurrencySymbol'] == "USD") & (men_currently_usd['ConvertedSalary'] < 200000) & (men_currently_usd['ConvertedSalary'] > 0) & (men_currently_usd['SalaryType'] == "Yearly")]

men_currently_usd.sort_values("ConvertedSalary", axis = 0, ascending = True, inplace = True) 

sample_men = men_currently_usd.sample(n=700, random_state=1)
sample_woman = women_currently_usd.sample(n=700, random_state=1)


# ### Adding the Gender Filter, Filtering by "1" which represents Males

# In[2188]:


male_participants_numeric_data = filtered_numeric_data.loc[filtered_numeric_data['q3Gender'] == "1"]
male_participants_numeric_data.head()


# ### Returning the size of the DataFrame now that only the Females are selected

# In[22]:


female_participants_numeric_data.shape[0]


# ### Returning the size of the DataFrame now that only the males are selected

# In[23]:


male_participants_numeric_data.shape[0]


#     

# In[26]:


filtered_female_participants_numeric_data.shape[0]


# In[27]:


# TO DO: MATCH THIS DATA WITH AGE CATEGORY
float_vals_age = filtered_female_participants_numeric_data.loc[:,"q2Age"].astype('float64')
round(float_vals_age.mean())


# In[28]:


filtered_female_participants_numeric_data.loc['q2Age'] = float_vals_age


# ## TIDY DATA FOR Numeric Mappping. Transposing and Grouping Data

# ### Transposing the Data to make it easier to read

# In[2185]:


transposed = numeric_mapping_data.transpose()


# In[2184]:


transposed.groupby(transposed.columns, axis=1).sum()


# ### Structuring the Data

# ### Merging the data from the two DataFrames on certain columns for Male and Female Participants 

# ### Females

# In[2186]:


u = female_participants_numeric_data
v = values_female_participants[['q2Age','CountryNumeric2']]
out = pd.concat([u, v], 1)
out.head()
list_of_columns = list(out.columns.values)
length_of_list = len(list_of_columns )
list_of_columns [length_of_list-2] = 'q2Age2'
list_of_columns [length_of_list-1] = 'CountryNumeric'
out.columns = list_of_columns 


# ### Males

# In[2191]:


u = male_participants_numeric_data
v = values_male_participants[['q2Age','CountryNumeric2']]
out2 = pd.concat([u, v], 1)
out2.head()
list_of_columns = list(out.columns.values)
length_of_list = len(list_of_columns )
list_of_columns [length_of_list-2] = 'q2Age2'
list_of_columns [length_of_list-1] = 'CountryNumeric'
out2.columns = list_of_columns 


# In[34]:


out.head()


# In[256]:


out['q2Age2'].head()


# In[2195]:


count_by_age = out.groupby('q2Age2').count()
count_by_age


# In[2196]:


count_by_age['q2Age']


# In[38]:


pip install plotnine


# # Exploratory Data Analysis

# In[2193]:


# import the plotnine package
import plotnine as p9


# ### Percentage of Females Present in Survey of data

# In[41]:


(values_data.loc[values_data['q3Gender'] == "Female"].shape[0]/values_data.shape[0])*100


# In[42]:


values_data.head()


# ### Sorting DataFrame According to Age that the person started to Code

# In[43]:


# Viewing Data in Ascending Order from when the person started to code
out.sort_values("q1AgeBeginCoding", axis=0, ascending=True, inplace=True)
female_values_data = out.loc[out['q3Gender'] == "2"]
male_values_data = out2.loc[out2['q3Gender'] == "1"] 


# In[2199]:


male_values_data.head()


# ### Showing Information about the Male and Female Ratio in Canada and India

# In[45]:


female_values_data.shape[0]


# In[46]:


count_by_country_female =female_values_data.groupby('CountryNumeric').count()
count_by_country_female.at['Canada','q3Gender']


# In[47]:


count_by_country_male =male_values_data.groupby('CountryNumeric').count()
count_by_country_male.at['Canada','q3Gender']


# In[48]:


count_by_country_female_in =female_values_data.groupby('CountryNumeric').count()
count_by_country_female_in.at['India','q3Gender']


# In[49]:


count_by_country_male_in =male_values_data.groupby('CountryNumeric').count()
count_by_country_male_in.at['India','q3Gender']


# ## Plotting with comparing India to Canada

# In[50]:


# import the plotnine package
import plotnine as p9
#x = count_by_age.plot.bar(y='CountryNumeric2', rot=90)

df = pd.DataFrame({'male': [count_by_country_male_in.at['India','q3Gender'],count_by_country_male.at['Canada','q3Gender']]
                    ,'female': [count_by_country_female.at['India','q3Gender'],count_by_country_female_in.at['Canada','q3Gender']]}, index=['India','Canada'])
ax = df.plot.bar(rot=0)


# ### Ratio of Males to Females in Canada and India

# In[51]:


total_indians = count_by_country_male_in.at['India','q3Gender']+count_by_country_female_in.at['India','q3Gender']
total_indians


# In[52]:


total_canadians = count_by_country_male.at['Canada','q3Gender']+count_by_country_female.at['Canada','q3Gender']


# In[53]:


count_by_country_male_in.at['India','q3Gender']/total_indians


# ## Cleaning the Data to split it by ;

# In[2200]:


test = women_stack_overflow['DevType'].str.split(';', expand=True)
test.head()


# ### Finding sizes of each category of developers

# In[187]:


women_count_full_stack = test[test.eq('Full-stack developer').any(1)].shape[0]
women_count_front_end = test[test.eq('Front-end developer').any(1)].shape[0]
women_count_back_end = test[test.eq('Back-end developer').any(1)].shape[0]


# In[2198]:


df = pd.DataFrame({'dev_type': [women_count_full_stack, women_count_front_end,women_count_back_end]},index=['Full-Stack', 'Front-End','Back-End'])
df 


# ## Graphing the counts for Front-end, Back-end and Full-Stack developers

# In[314]:


# Load Matplotlib and data wrangling libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import set_matplotlib_formats


plt.rcParams['figure.figsize'] = (8, 5)
fig, ax = plt.subplots()


set_matplotlib_formats('retina', quality=100)
bars = plt.bar(
    x=np.arange(df.size),
    height=df['dev_type'],
    tick_label=df.index
)

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

# Grab the color of the bars so we can make the
# text the same color.
bar_color = bars[0].get_facecolor()
for bar in bars:
    ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 50,
      round(bar.get_height(), 1),
      horizontalalignment='center',
      color='blue',
      weight='bold'
  )

# Make the chart fill out the figure better.
fig.tight_layout()


# In[164]:


women_clean_stackoverflow_language_worked_with = women_stack_overflow.dropna(subset=['DevType'])
women_clean_stackoverflow_language_worked_with.groupby('DevType').count()
#amount_for_each_gender = values_data.groupby('q3Gender').count()


# ### Only selecting data relevant to women

# In[3]:


total_data_stackoverflow['Gender'].head()


# In[4]:


total_data_stackoverflow.loc[total_data_stackoverflow['Gender'] == "Female"].shape


# In[5]:


total_data_stackoverflow.loc[total_data_stackoverflow['Gender'] == "Male"].shape


# In[2202]:


count_by_country_male_in =total_data_stackoverflow.groupby('Gender').count()
count_by_country_male_in['Respondent']['Male']
count_by_country_male_in.index
test= count_by_country_male_in.copy()
count_by_country_male_in=count_by_country_male_in.reset_index()

a = count_by_country_male_in.loc[count_by_country_male_in['Gender'] == "Female"]
b = count_by_country_male_in.loc[count_by_country_male_in['Gender'] == "Male"]

testing = count_by_country_male_in.columns
df1 = pd.DataFrame(columns=testing)
df1 = df1.append([a,b],ignore_index=True)
df1


# In[49]:


total_data_stackoverflow.head()


# In[2201]:


fm = total_data_stackoverflow.loc[total_data_stackoverflow['Gender'] == "Female"]
ml = total_data_stackoverflow.loc[total_data_stackoverflow['Gender'] == "Male"]
result = pd.merge(fm, ml, how='outer', on=['Respondent', 'Gender'])

result['Gender'].head()


# ## Viewing the distribution of Males and Females in the StackOverflow Dataset

# In[72]:


# the amount of counts per type of sex (gender)

(p9.ggplot(data=result, mapping = p9.aes(x='factor(Gender)',fill='factor(Gender)'))
    + p9.geom_bar()
 + p9.theme(text=p9.element_text(size=20))
)


# In[7]:


gooddata = df1.iloc[0:2]


# In[8]:


count_by_country_male_in.iloc[:,1]


# In[89]:


df1['Respondent'][0]


# ## Displaying Data Male Versus Female on Bar Graph StackOverflow

# In[47]:



import plotnine as p9

(p9.ggplot(data=df1, mapping = p9.aes(x='Gender',y="Respondent",fill='factor(Gender)'))
    + p9.geom_bar(stat="identity")
 +p9.scale_y_discrete()
  + p9.theme(text=p9.element_text(size=20))            

)


# ## Displaying a pie chart for the distribution of males vs females on stackoverflow

# In[107]:


df = pd.DataFrame({'count': [df1['Respondent'][0], df1['Respondent'][1]]},index=['Female', 'Male'])
plot = df.plot.pie(y='count', figsize=(5, 5),colors = ['pink', 'lightblue'],autopct='%1.1f%%')


# In[112]:


values_data.head()


# In[124]:


amount_for_each_gender = values_data.groupby('q3Gender').count()
amount_for_each_gender['RespondentID']['Male']


# ## Displaying a pie chart for the distribution of males vs females on HackerRank

# In[125]:


df = pd.DataFrame({'count': [amount_for_each_gender['RespondentID']['Female'], amount_for_each_gender['RespondentID']['Male']]},index=['Female', 'Male'])
plot = df.plot.pie(y='count', figsize=(5, 5),colors = ['pink', 'lightblue'],autopct='%1.1f%%')


# In[2180]:


fm = values_data.loc[values_data['q3Gender'] == "Female"]
ml = values_data.loc[values_data['q3Gender'] == "Male"]
genders = pd.merge(fm, ml, how='outer', on=['RespondentID', 'q3Gender'])
genders.head()


# ## Displaying a bar chart for the distribution of males vs females on HackerRank
# 

# In[134]:


(p9.ggplot(data=genders, mapping = p9.aes(x='factor(q3Gender)',fill='factor(q3Gender)'))
    + p9.geom_bar()
 + p9.theme(text=p9.element_text(size=20))
)


# # Adding Data for precise count of graph for HackerRank

# In[2203]:


total_data_stackoverflow.columns.values


# In[2178]:


total_data_stackoverflow['LanguageWorkedWith'].head()


# In[136]:


values_data.columns.values


# In[2177]:


values_data['q4Education'].head()


# In[ ]:


fm = total_data_stackoverflow.loc[total_data_stackoverflow['Gender'] == "Female"]


# In[2204]:


female_values_data = values_data.loc[values_data['q3Gender'] == "Female"]
female_values_data = female_values_data.dropna(axis=0, subset=['q4Education','q9CurrentRole'])
female_values_data.head()


# ## Showing the level of education relative to the job level 

# In[2205]:


(p9.ggplot(data = female_values_data,mapping = p9.aes(x='q4Education', y='q8JobLevel',color='q4Education'))
 + p9.geom_point(alpha=0.7)
 + p9.facet_wrap("q4Education")
)


# In[2207]:


unemployed_females = female_values_data.loc[female_values_data['q9CurrentRole'] == "Unemployed"]
unemployed_females.head()


# In[2208]:


un = female_values_data.loc[female_values_data['q9CurrentRole'] == "Unemployed"]
em = female_values_data.loc[(female_values_data['q9CurrentRole'] != "Unemployed") & (female_values_data['q9CurrentRole'] != "Student")]

# Create the total score for each group
totals = [un.shape[0],em.shape[0]]
totals

total_count_each_group =female_values_data.groupby('q4Education').count()
total_count_each_group['q9CurrentRole']


# In[2209]:


unemployed_education =un.groupby('q4Education').count()
unemployed_education['q9CurrentRole']


# In[2210]:


employed_education =em.groupby('q4Education').count()
employed_education['q9CurrentRole']
employed_education['q9CurrentRole'].values


# In[528]:


ppercent['employed'] = (employed_education['q9CurrentRole']/total_count_each_group['q9CurrentRole'])*100
ppercent['unemployed'] = (unemployed_education['q9CurrentRole']/total_count_each_group['q9CurrentRole'])*100
ppercent['unemployed'].values[0:-1]
#ppercent['employed'].values[0:-1]


# In[538]:


raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'pre_score': [4, 24, 31, 2, 3],
        'mid_score': [25, 94, 57, 62, 70],
        'post_score': [5, 43, 23, 23, 51]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'pre_score', 'mid_score', 'post_score'])
df


# ## Graphing Employment Rate and Unemployment Rate
# 

# In[768]:


# Create a figure with a single subplot
f, ax = plt.subplots(1, figsize=(15,10))

# Set bar width at 1
bar_width = 1

# positions of the left bar-boundaries
bar_l = [i for i in range(6)] 

# positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i+(bar_width/10) for i in bar_l] 

# Create the total score for each participant
#totals = [i+j+k for i,j,k in zip(df['pre_score'], df['mid_score'], df['post_score'])]
totals =total_count_each_group['q9CurrentRole']
totals = list(totals)
# Create the percentage of the total score the pre_score value for each participant was
#pre_rel = [i / j * 100 for  i,j in zip(df['pre_score'], totals)]
pre_rel = ppercent['employed'].values[1:-1]
pre_rel = list(pre_rel)
# Create the percentage of the total score the mid_score value for each participant was
#mid_rel = [i / j * 100 for  i,j in zip(df['mid_score'], totals)]
mid_rel = ppercent['unemployed'].values[1:-1]
mid_rel = list(mid_rel)
# Create the percentage of the total score the post_score value for each participant was
post_rel = [i / j * 100 for  i,j in zip(df['post_score'], totals)]

# Create a bar chart in position bar_1
ax.bar(bar_l, 
       # using pre_rel data
       pre_rel, 
       # labeled 
       label='Pre Score', 
       # with alpha
       alpha=0.9, 
       # with color
       color='#019600',
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Create a bar chart in position bar_1
ax.bar(bar_l, 
       # using mid_rel data
       mid_rel, 
       # with pre_rel
       bottom=pre_rel, 
       # labeled 
       label='Mid Score', 
       # with alpha
       alpha=0.9, 
       # with color
       color='#219AD8', 
       # with bar width
       width=bar_width,
       # with border color
       edgecolor='white'
       )

# Set the ticks to be first names
ex = ['College graduate','High school graduate' ,'Post graduate degree (Masters, PhD)',
      'Some college','Some high school','Some post graduate work (Masters, PhD)' ]                   
plt.xticks(tick_pos, ex )

ax.set_ylabel("Percentage")
ax.set_xlabel("")

# Let the borders of the graphic
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
plt.ylim(0, 65)

# rotate axis labels
plt.setp(plt.gca().get_xticklabels(), rotation=340, horizontalalignment='left')

# shot plot
plt.show()


# In[1843]:


totals


# In[2172]:


women_currently_usd = women_stack_overflow.dropna(axis=0, subset=['CurrencySymbol','YearsCodingProf'])
women_currently_usd = women_currently_usd.loc[(women_currently_usd['CurrencySymbol'] == "USD")& (women_currently_usd['ConvertedSalary'] < 200000) &(women_currently_usd['ConvertedSalary'] > 0) & (women_currently_usd['SalaryType'] == "Yearly") ]

women_currently_usd = women_currently_usd.sort_values(by='YearsCodingProf',ascending=True)
#women_currently_usd['YearsCodingProf']
#women_currently_usd['YearsCodingProf'] = '0' + women_currently_usd['YearsCodingProf'].astype(str)
#women_currently_usd['YearsCodingProf']
women_currently_usd['YearsCodingProf'] = pd.Categorical(women_currently_usd['YearsCodingProf'], ["0-2 years", "3-5 years","6-8 years","9-11 years","12-14 years","15-17 years","18-20 years"
                                                                                                 ,"21-23 years","24-26 years","27-29 years","30 or more years"])
women_currently_usd['YearsCodingProf']
women_currently_usd.sort_values("YearsCodingProf")['YearsCodingProf'].head()


# ## Showing Distribution of Salary with the Experience with years coding professionally

# In[921]:


(p9.ggplot(data = women_currently_usd,mapping = p9.aes(x='ConvertedSalary', y='YearsCodingProf'))
 
 + p9.theme(text=p9.element_text(size=5))
 + p9.geom_point()
)


# In[922]:



(p9.ggplot(data=women_currently_usd,
           mapping=p9.aes(x='YearsCodingProf',
                          y='ConvertedSalary'))
 + p9.theme(axis_text_x = p9.element_text(angle=90))
    + p9.geom_boxplot()
)


# In[832]:


values_female_participants


# In[870]:


pip install ggplot


# In[880]:



(p9.ggplot(data=values_female_participants,
           mapping=p9.aes(x='q1AgeBeginCoding',
                          y='q2Age',
                          color='q4Education'))
    + p9.geom_point(alpha=0.7)
    + p9.facet_wrap("q4Education") 
 + p9.stat_smooth(method='lm')
 + p9.theme(axis_text_x = p9.element_text(angle=90,size=8),axis_text_y = p9.element_text(size=8),text=p9.element_text(size=5))
           
 
 

)


# # Machine Learning and Matching

# ### First clean the Dataset so that Nan Values are removed

# In[1334]:


clean_converted_salary = total_data_stackoverflow.dropna(axis=0, subset=['ConvertedSalary','SalaryType','Currency'])
clean_converted_salary['ConvertedSalary'].head()
#total_data_stackoverflow['ConvertedSalary']


# In[1322]:


clean_gender = total_data_stackoverflow.dropna(axis=0, subset=['Gender'])
clean_gender['Gender'].head()
#total_data_stackoverflow['ConvertedSalary']


# ### Creating a new column in each dataset combining important attributes into one string (mixture), then use joins to view overlaps in values of important columns
# 

# In[2227]:


total_data_stackoverflow['mixture'] = clean_gender['Gender'] + ' '+ clean_converted_salary['ConvertedSalary'].astype('str')                                       +' '+clean_converted_salary['Currency']+' '+ clean_converted_salary['SalaryType']
total_data_stackoverflow['mixture'].head()


# In[2231]:


df = pd.DataFrame({'Gender':clean_gender['Gender'], 'Salary':clean_converted_salary['ConvertedSalary'].astype('str'), 
                   'Currency':clean_converted_salary['Currency'], 'Salary Type':clean_converted_salary['SalaryType'],
                  'Respondent':total_data_stackoverflow['Respondent']})

df = df.dropna(subset=["Salary", "Currency","Salary Type"], how='all')
female_values_data =df.loc[(df['Gender'] == "Female")]
df = df.loc[(df['Gender'] == "Female")]


# In[2232]:


df.head()


# In[2233]:


import py_entitymatching as em
em.set_key(df, 'Respondent')   # specifying the key column in the kaggle dataset


# ### Labelling of the Sample Data  (n=200)

# In[2234]:


# Sampling 100 pairs and writing this sample into a .csv file
sampled_five_hundred = female_values_data.sample(200, random_state=0)
sampled_five_hundred.to_csv('./datasets/f1.csv', encoding='utf-8')


# In[2235]:


labeled = em.read_csv_metadata('./datasets/f1.csv', ltable=df, rtable=df,
            fk_ltable='Respondent', fk_rtable='Respondent', key='Respondent')
labeled.head()


# ### Train Machine Learning Algorithms

# In[2237]:


split = em.split_train_test(labeled, train_proportion=0.5, random_state=0)


train_data = split['train']
test_data = split['test']
dt = em.DTMatcher(name='DecisionTree', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')
nb = em.NBMatcher(name='NaiveBayes')
knn = em.NBMatcher(name='knn')



# In[2238]:


attr_corres = em.get_attr_corres(female_values_data, female_values_data)
attr_corres['corres'] = [('Respondent', 'Respondent'), 
('Gender', 'Gender'),
('Salary Type', 'Salary Type'),
('Currency', 'Currency'),]

l_attr_types = em.get_attr_types(female_values_data)
r_attr_types = em.get_attr_types(female_values_data)

tok = em.get_tokenizers_for_matching()
sim = em.get_sim_funs_for_matching()

F = em.get_features(female_values_data, female_values_data, l_attr_types, r_attr_types, attr_corres, tok, sim)
F.head()


# In[2225]:


train_data.head()


# In[2126]:


#train_features = em.extract_feature_vecs(train_data['Label '], feature_table=F, attrs_before='Gender', show_progress=False) 
train_features = em.impute_table(train_data,exclude_attrs=['Respondent','Gender','Salary Type','Currency'], strategy='mean')


# In[2212]:


train_features.head()


# In[2153]:



# Fixing issue with metric here
result = em.select_matcher([dt, rf,lg, ln,nb,knn], table=train_features, 
                       exclude_attrs=['Respondent','Gender','Salary Type','Currency'], k=5,target_attr='Label', random_state=0)
#result['cv_stats']
# filtering what is written below instead of result['cv_stats']
result['drill_down_cv_stats']['f1']



# In[2150]:


test_features.head()


# # Choosing DT (Decision Tree) as the Algorithmn for Prediction 

# In[2151]:


best_model = result['selected_matcher']
best_model


# In[2239]:


train_data.head()


# In[2152]:


best_model = result['selected_matcher']
best_model.fit(table=train_features, exclude_attrs=['Respondent','Gender','Salary Type','Currency'], target_attr='Label')

#test_features = em.extract_feature_vecs(test_data, feature_table=F, attrs_after='Label ', show_progress=False)
train_features = em.impute_table(train_data,exclude_attrs=['Respondent','Gender','Salary Type','Currency'], strategy='mean')
test_features = em.impute_table(test_data,exclude_attrs=['Respondent','Gender','Salary Type','Currency'], strategy='mean')
test_features
# Predict on the test data
predictions = best_model.predict(table=test_features, exclude_attrs=['Gender','Respondent','Currency','Salary Type','Unnamed: 0'], append=True, target_attr='predicted', inplace=False)
predictions

# Evaluate the predictions
eval_result = em.eval_matches(predictions, 'Label', 'predicted')
em.print_eval_summary(eval_result)


# In[2211]:


predictions.head()


# # Linear Regression Model

# ## Using this model to create a relation between a man's salary (x-axis) and woman's salary (y-axis). 

# ## Using the best fit line, one can predict the salary of the opposite gender 

# In[1292]:


sample_woman.sort_values("ConvertedSalary", axis = 0, ascending = True, inplace = True) 
sample_woman['ConvertedSalary']

sample_men.sort_values("ConvertedSalary", axis = 0, ascending = True, inplace = True) 

sample_men['ConvertedSalary']

men = sample_men['ConvertedSalary'].values
women = sample_woman['ConvertedSalary'].values


import matplotlib.pyplot as plt
import pandas as pd

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b
    
# solution
a, b = best_fit(men,women)
plt.scatter(men, women, color='c')
yfit = [a + b * xi for xi in men]
plt.plot(men, yfit,color='r')
#plt.scatter(grades_range, men_wages, color='g')
plt.xlabel('Men Wages')
plt.ylabel('Women Wages')
plt.show()


# In[2173]:


women = sample_woman['ConvertedSalary'].values
women.mean()


# In[2175]:


men = sample_men['ConvertedSalary'].values
men.mean()


# ## Correlation

# In[1247]:


np.corrcoef(men, women)


# ## Box Plot for Men V.s. Women's Salary

# In[2176]:


import matplotlib.pyplot as plt
 

box_plot_data=[men,women]
plt.boxplot(box_plot_data)
plt.xticks([1, 2], ['Men', 'Women'])
plt.show()


# ### Age Distribution of When Women Start To Code

# ### Filtering by numeric data values

# In[1855]:


female_values_data['q1AgeBeginCoding'] = female_values_data['q1AgeBeginCoding'].replace("#NULL!",np.nan)


# In[1856]:


female_values_data = female_values_data.dropna(subset=['q1AgeBeginCoding'])


# In[1857]:


female_values_data.shape[0]


# In[1859]:


female_values_data['q1AgeBeginCoding'] = pd.Categorical(female_values_data['q1AgeBeginCoding'], ["5 - 10 years old", "11 - 15 years old","16 - 20 years old","21 - 25 years old","26 - 30 years old","31 - 35 years old","36 - 40 years old","41 - 50 years old","50+ years or older"])


# ### Age Distribution of When Women Start To Code

# In[1860]:


(p9.ggplot(data= female_values_data,
           mapping=p9.aes(x='factor(q1AgeBeginCoding)'))
    + p9.geom_bar()
 + p9.theme(axis_text_x = p9.element_text(angle=90))
)
#female_values_data = values_data[values_data['q3Gender'] == "Female"]


# ## Age Distribution of When Men Start To Code

# In[1983]:


male_values_data['q1AgeBeginCoding'] = male_values_data['q1AgeBeginCoding'].replace("#NULL!",np.nan)


# In[1985]:


male_values_data = male_values_data.dropna(subset=['q1AgeBeginCoding'])


# ### Re-ordering the labels to make them appear from youngest age group to oldest

# In[1987]:


male_values_data['q1AgeBeginCoding'] = pd.Categorical(male_values_data['q1AgeBeginCoding'], ["5 - 10 years old", "11 - 15 years old","16 - 20 years old","21 - 25 years old","26 - 30 years old","31 - 35 years old","36 - 40 years old","41 - 50 years old","50+ years or older"])


# ### Age Distribution of When Men Start To Code

# In[1988]:


(p9.ggplot(data= male_values_data,
           mapping=p9.aes(x='factor(q1AgeBeginCoding)'))
    + p9.geom_bar()
 + p9.theme(axis_text_x = p9.element_text(angle=90))
)


# # Storytelling
# ## As an overview, this analysis gave a better idea on how women are involved in tech. It showed how inequalities still exist for women in technology as their male counterparts get paid more than them which is not only an issue in the STEM field but also in the workforce in general. There is a 10% disparity with what they are paid compared to what men are paid. The linear regression model can give a better idea of how payment compares at a certain value for both men and women.  
# 

# ## At What Age do Women compared to Men Start Learning how to code ? Why this disparity?
# ## Women generally learn to code later in life then men who start to learn how to code at a relatively younger range (11-15 years mostly).Women typically learn to code at a later age than their male counterparts as the graph shows that a greater proportion of men learn to code from 11-15 years old than women. This may be caused by the fact that these early pre-teen and teenage years women may be influenced by gender stereotypes to steer away from male dominated interests like STEM related technologies such as learning how to code. From 16-20 years old, men and women are equally proportional in terms of learning how to code then. This may be due to the fact that around this age, females are preparing to go to college or start their first years in college. Since college is a more liberating environment for people in general as the social pressures of high school being gone, women might be more prone to try something new without the fear of being judged (aka learning how to code) and pursuing STEM related majors. 

# ## What type of technologies do women prefer to code in ? Do they prefer Front-end / Back-end or Full-Stack? What Coding Languages are most popular for Women? 

# ## Women mostly prefer working with back-end technologies as well as having their preferred languages being Java and JavaScript. Full-Stack Coding and Front-end Coding are pretty similar in prefereneces for women

# ## What level of education is optimal for women to have in order for them to reduce their level of unemployment?

# ## According to this graph, if women want to be ensured that they are the least likely to be unemployed, they should do only some college or finish college and only do some post graduate work and not complete their post graduate degree.As expected, if a female does not finish high school, she will most likely be unemployed as employers usually want their employees to have some level of basic education   
# 

# In[ ]:




