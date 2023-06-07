#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load data
gsearch_jobs=pd.read_csv('D:\Sales Data analysis python\gsearch_jobs.csv')


# In[3]:


#droping irrelevant column
gsearch_jobs.drop("Unnamed: 0",axis=1,inplace=True)


# In[4]:


#show dataframe
gsearch_jobs.head(3)


# In[5]:


#checking Datatype of DataFrame gsearch_jobs and not null values present in each column 
gsearch_jobs.info()


# In[6]:


#List features of gsearch_jobs and calculate the associated number of missing values per feature
gsjob_of_null_values = gsearch_jobs.isnull().sum()
for k, v in gsjob_of_null_values.items():
    percentage = round((v * 100 / gsearch_jobs['index'].index.size),2)
    print(k,", ",v, "(", percentage ,"%)")


# # Notes for gsearch_jobs dataset:
# 
# Feature location has 0.08 % missing value 
# feature salaries has more than 80% missing value which is huge here we will fill blank salary value with average salaries 
# according to respective job_type to get more accuracy in data

# In[7]:


gsearch_jobs.head(2)


# In[8]:


#Converting salary_min,salary_max into per annum for monthly and hourly data as per salary rate
gsearch_jobs.loc[gsearch_jobs['salary_rate'] == 'a month', ['salary_min','salary_max']] *= 12
gsearch_jobs.loc[gsearch_jobs['salary_rate'] == 'an hour', ['salary_min','salary_max']] *= 2080
gsearch_jobs.loc[gsearch_jobs['salary_rate'] == 'a year', ['salary_min','salary_max']] *=1


# In[9]:


gsearch_jobs.tail(2)


# In[10]:


#selecting only required and useful columns
gsearch_jobs=gsearch_jobs[['index','title','company_name','location','via','description','extensions','job_id','thumbnail',
'posted_at','schedule_type','work_from_home','search_term','date_time','search_location',
'salary_min','salary_max','salary_standardized','description_tokens']]


# In[11]:


gsearch_jobs.head(2)


# In[12]:


# Calculate average salaries for each schedule_type and create DataFrames
avg_salary_min = pd.DataFrame(gsearch_jobs.groupby(['schedule_type'])['salary_min'].mean()).reset_index()
avg_salary_max = pd.DataFrame(gsearch_jobs.groupby(['schedule_type'])['salary_max'].mean()).reset_index()
avg_salary_std = pd.DataFrame(gsearch_jobs.groupby(['schedule_type'])['salary_standardized'].mean()).reset_index()

# Merge the DataFrames on 'schedule_type'
merged_df = gsearch_jobs.merge(avg_salary_min, on='schedule_type', how='left')
merged_df = merged_df.merge(avg_salary_max, on='schedule_type', how='left')
merged_df = merged_df.merge(avg_salary_std, on='schedule_type', how='left')

# Fill missing values in 'salary_min', 'salary_max', and 'salary_standardized' with respective average values
merged_df['salary_min_x'] = merged_df['salary_min_x'].fillna(merged_df['salary_min_y'])
merged_df['salary_max_x'] = merged_df['salary_max_x'].fillna(merged_df['salary_max_y'])
merged_df['salary_standardized_x'] = merged_df['salary_standardized_x'].fillna(merged_df['salary_standardized_y'])

# Drop the average columns
merged_df = merged_df.drop(['salary_min_y', 'salary_max_y', 'salary_standardized_y'], axis=1)


# In[13]:


merged_df.head(3)


# In[14]:


#Rrenaming the salaries column
merged_df.rename(columns={'salary_min_x':'salary_min','salary_max_x':'salary_max','salary_standardized_x':'salary_standardized'},inplace=True)


# In[15]:


merged_df.info()


# # Percentage of schedule/Job types offered by companies 
# 

# In[16]:


from matplotlib import pyplot as plt
plt.figure(figsize=(7.5,8.5))
emp_type=merged_df['schedule_type'].value_counts(dropna=True)
myexplode = [0.05, 0.03, 0.2, 0.4]
plt.pie(emp_type,labels=emp_type.index,autopct='%1.1f%%',explode=myexplode)
plt.axis('equal')
plt.title('Distribution of Schedule Types',fontsize=18,fontweight='bold')

legend_labels = []
for i, count in enumerate(emp_type):
    if pd.isna(emp_type.index[i]):
        label = 'Blank'
    else:
        label = emp_type.index[i]
    legend_labels.append(f"{label}: {count}")

# Add legends based on the label list
legend_title = 'Employment Types (Count)'
legend=plt.legend(legend_labels, loc='upper right', title=legend_title)
legend.get_title().set_fontweight('bold')

plt.show()
#here we can say that companies mostly hires for full time and after that a big percentage of companies also advertize recruitment on contract basis


# # Most used Job sites for posting the job recrutiment 
# 

# In[17]:


#count the number of advertisement posted on job recruting sites 
gsearch_via_count=pd.DataFrame(merged_df['via'].value_counts(dropna=True))
gsearch_via_count=gsearch_via_count.sort_values('count',ascending=False)
#For better visualization just consider top 10 most used sites
gsearch_via_count=gsearch_via_count.head(10)
gsearch_via_count.index = gsearch_via_count.index.str.replace('via', '')


# In[18]:


gsearch_via_count


# In[19]:


plt.figure(figsize=(12,6))
plt.bar(gsearch_via_count.index,gsearch_via_count['count'])
plt.xlabel('job advertisement via',fontsize=16)
plt.ylabel('number of Job posted',fontsize=16)
plt.title('Top 10 Recruiter Sites',fontsize=18,fontweight='bold')
plt.show
#from here we can see most popular job recruitment sites is linkedln and then upwork and bee 
#but linkedln is far away from all other recruitment sites


# In[20]:


merged_df['location'].nunique()


# 
# # top 10 Companies which is providing max avgsalary

# In[21]:


merged_df.dtypes


# In[22]:


#changing following columns to float
merged_df[['salary_min','salary_max']]=merged_df[['salary_min','salary_max']].astype('float')


# In[23]:


#finding the maximum avg salary by companies
gsearch_jobs_avg_sal=merged_df.groupby('company_name').agg({
    'salary_min': 'mean',
    'salary_max': 'mean',
    'salary_standardized':'mean'
})
#Top 10 companies offering maximum salary_max
top_10_salary_max=gsearch_jobs_avg_sal[['salary_max']]
top_10_salary_max=top_10_salary_max.sort_values('salary_max',ascending=False).head(10)


# In[24]:


#plot a horizontal bar graph
from matplotlib import pyplot as plt
plt.figure(figsize=(10,29))
ax=top_10_salary_max.plot.barh(y='salary_max',legend=False)
for index, value in enumerate(top_10_salary_max['salary_max']):
    ax.text(value + 1, index, str(value), color='black')
# Set the plot title and labels
ax.set_title('Top Companies offering highest maximum average salary',fontweight='bold')
ax.set_xlabel('salary--------->')
ax.set_ylabel('Company Name--------->')
#here we can see that the Corps team offering higesht maximum salary_max i.e. 428480 and it is far way from others companies 


# # Draw a compare graph for Standard salary, avg max and min offered salary by Schedule type 
# 

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt

# Group the data by schedule_type and calculate the average salary
average_df = merged_df.groupby('schedule_type')[['salary_min', 'salary_max','salary_standardized']].mean()

# Reset the index to make 'company' a regular column
average_df.reset_index(inplace=True)
plt.plot(average_df['schedule_type'], average_df['salary_min'], marker='o', label='Max Salary')
plt.plot(average_df['schedule_type'], average_df['salary_max'], marker='o', label='Min Salary')
plt.plot(average_df['schedule_type'], average_df['salary_standardized'], marker='o', label='Average Salary')

# Customizing the plot
plt.xlabel('Schedule Type')
plt.ylabel('Salary')
plt.title('Salary Distribution by Schedule Type',fontweight='bold')
legend_labels=['salary_min', 'salary_max','salary_standardized']
plt.legend( legend_labels)


# Display the plot
plt.show()

#From here we can conclude that the internship having less than half average salary of full time, part time 
# or contractor, and contractor  basis job salary is almost equal to full time salary and part time job salary
#so since in part time job is having less working hours and paid amount per month is also good


# # Show the average salary based on Geographical map

# In[26]:


new_merged_df=merged_df.dropna(subset=['location'])
#remove all location columns which contains United States beacuse we are analysing statewise
new_merged_df=new_merged_df[~new_merged_df['location'].str.contains('United States', case=False)]
#str.split will split it into two string based on comma and str[1] will give the second string and str.strip()
#would remove leading or trailing spaces and it will give state code
new_merged_df['location'] = new_merged_df['location'].str.split(',').str[1].str.strip()
#it will remove the values that contains '(' 
new_merged_df['location'] = new_merged_df['location'].str.split('(').str[0].str.strip()
#drop those rows which having na in column location
new_merged_df=new_merged_df.dropna(subset=['location'])


# In[27]:


#finding the unique statecode
state_code=pd.DataFrame(new_merged_df['location'].unique())
state_code.rename(columns={0:'location'},inplace=True)


# In[28]:


pip install geopy


# In[29]:


pip install us


# In[30]:


import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from us import states

# Initialize geocoder
geolocator = Nominatim(user_agent="my-app")

# State code to state name mapping
state_mapping = {state.abbr: state.name for state in states.STATES}

# Function to get latitude and longitude
def get_coordinates(location):
    state_name = state_mapping.get(location)
    if state_name:
        location = geolocator.geocode(state_name + ", USA")
        if location:
            return location.latitude, location.longitude
    return None, None

# Add 'State' column
state_code['State'] = state_code['location'].map(state_mapping)

# Apply the function to the 'location' column
state_code[['Latitude', 'Longitude']] = state_code['location'].apply(get_coordinates).apply(pd.Series)


# In[31]:


#merge the state_code table to new_merged_df for refercing the state_name
new_merged_df=new_merged_df.merge(state_code,on='location')


# In[32]:


new_merged_df.columns


# In[33]:


#creating a dataframe map_merged_df to import necessary column in it to visualize the data on geospatial
map_merged_df=new_merged_df[['location','State','Latitude','Longitude','salary_standardized']]
map_merged_df.dropna(inplace=True)
map_merged_df = map_merged_df.groupby(['State','location', 'Latitude', 'Longitude'])[['salary_standardized']].mean().reset_index()

# Create a new dataframe with the desired columns
map_merged_df = pd.DataFrame({
    'location': map_merged_df['location'],
    'State':map_merged_df['State'],
    'Latitude': map_merged_df['Latitude'],
    'Longitude': map_merged_df['Longitude'],
    'salary_standardized': map_merged_df['salary_standardized']
})


# In[34]:


map_merged_df


# In[35]:


#using geopandas to converts lat and long to points
import geopandas as gpd
map_merged_df_geo=gpd.GeoDataFrame(map_merged_df,geometry=gpd.points_from_xy(map_merged_df.Longitude,map_merged_df.Latitude))
map_merged_df_geo


# In[38]:


pip install plotly


# In[39]:


import geopandas as gpd
#it will show the natural earth map
world_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# In[40]:


import plotly.express as px

fig = px.choropleth(
    map_merged_df_geo,
    locationmode='USA-states',
    locations='location',
    scope='usa',
    color='salary_standardized',   
    color_continuous_scale='Viridis',   
    labels={'average_salary': 'Average Salary'},  
)
# Add location annotations
for i, row in map_merged_df_geo.iterrows():
    fig.add_annotation(
        x=row['Longitude'],  
        y=row['Latitude'],  
        text=row['location'],  
        showarrow=False,
        font=dict(size=8),
    )

fig.update_layout(
    title_text='Average Salary by State',
    geo=dict(
        scope='usa',
        projection_type='albers usa',
        showlakes=True,  # You can set this to False if you don't want to show lakes
        lakecolor='rgb(255, 255, 255)',
    )
)

fig.show()
#Show from here we can say that CA i.e, California having higest average salary provider for Data Analyst salary in
#among states and Nebraska NE is providing min average salary in among states which less than half the reason behind it


# In[41]:


new_merged_df[new_merged_df['location']=='NE']
new_merged_df['salary_standardized'].min()
#So here we can see for NE i.e, Nebraska.we have only one opening so for that company is providing very less average salary
#even it is full time though they are providing less salary even it is min salary is provided by any designation


# # Job Opening per State/Location 

# In[42]:


#here spliting the location by comma to get the exact state associated with that
import pandas as pd

# Assuming you have a DataFrame called 'statewise' with a 'location' column
statewise = pd.DataFrame()
statewise['location']=merged_df['location']
# Define a function to extract the second string after a comma
def extract_second_string(value):
    if isinstance(value, str):
        if ',' in value:
            return value.split(',')[1].strip()
    return str(value)

# Apply the function to the 'location' column
statewise = pd.DataFrame(statewise['location'].apply(extract_second_string))


# In[43]:


# Remove leading/trailing spaces from all columns in the DataFrame
statewise =pd.DataFrame(statewise.applymap(lambda x: x.strip() if isinstance(x, str) else x))

# Display the updated DataFrame
statewise['location'].unique()


# In[44]:


import pandas as pd

# Assuming you have two DataFrames: statewise and state_code

# Merge the tables based on the 'location' column

statewise_merge = pd.merge(statewise, state_code, on='location', how='left')


# In[45]:


statewise_merge['location'].unique()


# In[46]:


#where ever state is na replace those with the location 
statewise_merge['State'].fillna(statewise_merge['location'], inplace=True)


# In[47]:


#wherever State is na fill that will be shown as Not available
statewise_merge['State']=statewise_merge['State'].replace('nan','Not available')


# In[48]:


statewise_merge['State'].unique()


# In[49]:


#replace United States (+ occupied string values to United States+ and 'KS   (+2 others)' as 'KS')
def modify_country(value):
    if "United States" in value:
        if "(+" in value:
            return value.split("(+")[0].strip() + "+"
        else:
            return value.strip()+'+'
    elif 'KS' in value:
        return value.split("(+")[0].strip()
    else:
        return value

# Apply the function to the 'State' column
statewise_merge['State'] = statewise_merge['State'].apply(modify_country)


# In[50]:


statewise_merge['State'].unique()


# In[51]:


#replacing KS location to Kansas
statewise_merge.loc[statewise_merge['State'] == 'KS', 'State'] = 'Kansas'


# In[52]:


statewise_merge['State'].unique()


# In[53]:


import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.hist(statewise_merge['State'], bins=len(statewise_merge['State'].unique()), edgecolor='black', alpha=0.7)

# Customizations
plt.xlabel('Location', fontweight='bold', fontsize=12)
plt.ylabel('Number of Openings', fontweight='bold', fontsize=12)
plt.title('Distribution of Locations', fontweight='bold', fontsize=14)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Adjust spacing to prevent label overlapping

plt.show()
#here we can see the highesh job opening advertized location for anywhere then it was in United State location and 
#in states Missouri has highest number of job opening 


# # Most Demanding skill for Data Analyst role

# In[54]:


#Just considering description_tokens column because we will count number of times it's ask for same skills
job_des=pd.DataFrame(merged_df['description_tokens'])


# In[55]:


#removing those column which is having '[]' in descrptions
job_des = job_des[~(job_des['description_tokens'] == '[]')]


# In[56]:


#droping na values 
job_des.dropna(inplace=True)


# In[57]:


import pandas as pd

# Split the values in the 'description_tokens' column into separate columns by comma and removing '[]'
job_des['description_tokens'] = job_des['description_tokens'].str.strip("[]").str.replace("'", "").str.split(", ")

# Find the maximum number of columns across all rows
max_columns = job_des['description_tokens'].apply(len).max()

# Create column names for the expanded columns
column_names = [f'description_tokens{i}' for i in range(1, max_columns + 1)]

# Expand the split values into separate columns
job_des = job_des.join(pd.DataFrame(job_des['description_tokens'].to_list(), columns=column_names))

# Drop the original 'description_tokens' column
job_des = job_des.drop('description_tokens', axis=1)


# In[58]:


import pandas as pd
import numpy as np

# Assuming you have a DataFrame named 'description_tokens' representing a table

# Iterate over each column
for column in job_des.columns:
    # Replace NaN values with a temporary value
    job_des[column] = job_des[column].fillna(np.nan)

    # Check if column values contain 'sql' and update them, ignoring rows with NaN
    job_des.loc[job_des[column].str.contains('sql', na=False), column] = 'sql'


# In[59]:


import pandas as pd
import numpy as np

# Assuming your data is stored in a DataFrame named 'job_des' with 21 columns containing job description_tokens(skills) names

# Calculate the total count for each skill across all columns
total_counts = {}
for column in job_des.columns:
    counts = job_des[column].str.lower().value_counts(dropna=False).to_dict()
    for skill, count in counts.items():
        if pd.notnull(skill):
            total_counts[skill] = total_counts.get(skill, 0) + count

# Convert the dictionary to a pandas Series for easy manipulation
total_counts_series = pd.Series(total_counts)
# Display the total counts for all skills
print(total_counts_series)


# In[60]:


#converting total_counts_series to DataFrame
total_counts_series=pd.DataFrame(total_counts_series)


# In[61]:


#renaming column name 
total_counts_series.rename(columns={0:'Count of skills in job description'},inplace=True)


# In[62]:


#top 10 demanding skills 
total_counts_series = total_counts_series.sort_values('Count of skills in job description',ascending=False)
total_counts_series=total_counts_series.head(10)


# In[63]:


job_des.shape
#so here we have 15581 jobs for which data we are performing


# In[64]:


import matplotlib.pyplot as plt

# Assuming you have defined the 'total_counts_series' variable with the data

total_counts_percentage = (total_counts_series / 15581) * 100

# Define a custom color palette
color_palette = ['#FFC107', '#3F51B5', '#4CAF50', '#F44336', '#9C27B0']

plt.figure(figsize=(10, 6))
total_counts_percentage.plot.bar(legend=False, color=color_palette)
plt.xlabel('Skills', fontweight='bold')
plt.ylabel('Percentage of Job Openings Demanding', fontweight='bold')
plt.title('Percentage of Skills in Job Description', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Display the chart
plt.show()
#here we can see the most demanding skill for a data analyst is sql around 60 % of companies asking for this skill as
# well as excel is second most demanding skill for same then tableau, python and power_bi is also demanding skills 


# # top 5 job advertising sites and their average salary 

# In[65]:


#since in gsearch_via_count already top 10 sites are there so taking head count of 5 for top 5 recruiters
top_5_sites=gsearch_via_count.head(5)
top_5_sites['via']=top_5_sites.index
top_5_sites = top_5_sites.reset_index(drop=True)


# In[66]:


top_5_sites = top_5_sites.reset_index(drop=True)


# In[67]:


sites_merged_df=merged_df
sites_merged_df['via']=sites_merged_df.loc[merged_df['via'].str.contains('via'), 'via'].str.replace('via', '')


# In[68]:


merge_sites_top_5 = sites_merged_df.merge(top_5_sites, on='via')


# In[69]:


merge_sites_top_5.dropna(subset=['salary_standardized'], inplace=True)


# In[70]:


merge_sites_top_5.info()


# In[71]:


import matplotlib.pyplot as plt

# Group the DataFrame by the 'via' column and calculate the average salary
av_sal = merge_sites_top_5.groupby('via')['salary_standardized'].mean()

# Define custom colors for the bar chart
colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']

# Plot the average salary using a bar chart
plt.figure(figsize=(10, 6))
av_sal.plot(kind='bar', color=colors, edgecolor='black')

# Add gridlines
plt.grid(axis='y', linestyle='--')

# Set labels and title
plt.xlabel('Job Sites', fontweight='bold')
plt.ylabel('Average Salary', fontweight='bold')
plt.title('Average Salary by Sites', fontweight='bold')

# Remove the top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Customize tick labels
plt.xticks(rotation=45, ha='right')
plt.tick_params(axis='x', labelsize=8)

# Display the chart
plt.tight_layout()
plt.show()

#From here we can say job site Upwork provides less average salary compare to othere and apart from that there is
#less differences in average salaries of all job advertising sites 


# # Conclusion:

# Based on our thorough analysis of the gsearch_jobs dataset, we have gathered some key insights
# that can help us better understand the job landscape for Data Analysts. Here's a summary of our
# findings:
# 1. Distribution of Job/Schedule Types:
# When it comes to job types, we noticed that most companies prefer hiring candidates for fulltime positions. However, we also observed a significant number of job openings for contractbased roles.
# 
# 2. Most Popular Job Recruitment Sites:
# LinkedIn emerged as the most widely used platform for job recruitment among the sites we
# analyzed. Upwork and Bee also had a notable presence, although LinkedIn maintained a clear
# lead.
# 
# 3. Companies Offering Maximum Average Salary:
# During our analysis, we identified the Corps team as the top payer, offering an impressive
# maximum salary of 428,480. This figure sets them apart from other companies in terms of
# compensation.
# 
# 4. Salary Disparity Across Job/Schedule Types:
# Our analysis revealed that internships tend to have an average salary that is less than half of
# what full-time, part-time, or contractor positions offer. However, contractor roles showed a
# salary level on par with full-time positions, while part-time positions had a slightly lower average
# salary. This suggests that part-time jobs can provide decent monthly pay despite fewer working
# hours.
# 
# 5. Geographical Variation in Average Salaries:
# When examining average salaries across different states, we found that California (CA) offers the
# highest average salary for Data Analyst positions. On the other hand, Nebraska (NE) provides
# the lowest average salary, which is less than half of what other states offer.
# 
# 6. Job Openings per State/Location:
# The United States as a whole has the highest number of job openings. Within the states,
# Missouri stood out with a significant number of advertised job opportunities.
# 
# 7. Most In-Demand Skills for Data Analysts:
# Based on our analysis, SQL emerged as the most sought-after skill, with approximately 60% of
# companies emphasizing its importance. Excel ranked second in demand, followed by Tableau,
# Python, and Power BI.
# 
# 8. Analysis of Job Advertising Sites:
# We observed that Upwork tends to provide a relatively lower average salary compared to other
# job advertising sites. However, the differences in average salaries among all the platforms were
# relatively minor.
# 
# These insights can serve as a helpful guide for individuals pursuing a career as a Data Analyst,
# providing valuable information about job trends and skill requirements. It's important to note that
# these findings are based on the analysis of the gsearch_jobs dataset and should be further
# explored and validated for more precise interpretations.
