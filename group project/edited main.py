# %% [markdown]
# ### Import some necessary libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %% [markdown]
# ### Import data

# %%
data1 = pd.read_csv('/Users/zihaopeng/vs_code/NTU-bc2406/dataset/application_record.csv')
data2 = pd.read_csv('/Users/zihaopeng/vs_code/NTU-bc2406/dataset/credit_record.csv')

# %% [markdown]
# ### Data overview

# %%
data1.shape

# %%
data2.shape

# %%
data1.head()

# %%
data2.head()

# %%
data1.info()

# %%
data2.info()

# %%
# Quickly view key statistical features of the data, understand the distribution of the data, 
# identify outliers or data entry errors, and more
pd.set_option('display.float_format', '{:.2f}'.format) 
data1.describe().T

# %%
data2.describe().T

# %% [markdown]
# ## Application Record Data Analyze and Cleaning

# %%
# Classification variables
class_cols = ['FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CODE_GENDER', 'FLAG_OWN_REALTY', 'FLAG_OWN_CAR', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
# Continuous variables
conti_cols = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
# Discrete variables
discrete_cols = ['CNT_CHILDREN', 'CNT_FAM_MEMBERS']

# %%
# Create a subplot grid with 1 row and a number of columns equal to the number of discrete features
# Set the figure size to 10x5 inches and the DPI to 200 for high resolution
fig, axes = plt.subplots(1, len(discrete_cols), figsize=(1075/100, 605/100), dpi=200)

# Set the main title for the entire figure
fig.suptitle('Count Plot for Discrete Features', fontsize=16, fontweight='bold')

# Loop through each axis object and the corresponding discrete feature
for ax, variable in zip(axes, discrete_cols):
    # Create a count plot for the discrete feature on the current axis
    sns.countplot(x=variable, data=data1, ax=ax)
    
    # Set the title of the subplot to the name of the discrete feature
    ax.set_title(variable)
    
    
    # Display count labels on each bar in the count plot
    for p in ax.patches:
        # Get the height of the current bar
        height = p.get_height()
        # Annotate the bar with its height value, placing the text in the center of the bar
        ax.annotate(f'{height}', xy=(p.get_x() + p.get_width() / 2, height), 
                    ha='center', va='bottom')
        

# Adjust the layout to ensure that subplots fit within the figure area
plt.tight_layout()
plt.savefig('plot3.png', dpi=200)
plt.show()

# %%
# Create a subplot grid with a number of rows equal to the number of continuous features
# and 2 columns for box plots and histograms
fig, axes = plt.subplots(len(conti_cols), 2, figsize=(1075/100, 605/100), dpi=200)

# Set the main title for the entire figure
fig.suptitle('Box plots and Histograms for Continuous Features', fontsize=16, fontweight='bold')

# Loop through each continuous feature and its corresponding subplot row index
for i, variable in enumerate(conti_cols):
    # Create a box plot for the current feature on the left column
    sns.boxplot(x=data1[variable], ax=axes[i, 0])
    axes[i, 0].set_title(f'Box plot of {variable}')
    
    # Create a histogram for the current feature on the right column, with KDE overlay
    sns.histplot(data1[variable], ax=axes[i, 1], kde=True)
    axes[i, 1].set_title(f'Histogram of {variable}')

# Adjust the layout to ensure that subplots fit within the figure area
plt.tight_layout()
plt.savefig('plot1.png', dpi=200)
plt.show()

# 年收入的数量，出生日数，负值表示距今天数，工作天数

# %% [markdown]
# From the above figure, it is found that the distribution of DAYS_EMPLOYED is very abnormal, so the serious outly value is filtered and eliminated.

# %%
data1[data1['DAYS_EMPLOYED'] > 0]['DAYS_EMPLOYED'].unique()

# %%
data1[data1['DAYS_EMPLOYED'] == 365243].shape

# %%
data1 = data1[data1['DAYS_EMPLOYED'] != 365243]
data1.shape

# %% [markdown]
# From the above figure, it is found that the distribution of AMT_INCOME_TOTAL is also abnormal, so the outly value is filtered.

# %%
# > 75th percentile
data1[data1['AMT_INCOME_TOTAL'] > 2250000].shape

# %%
data1[data1['AMT_INCOME_TOTAL'] > 600000.0].shape[0]

# %%
fig, ax = plt.subplots(1, 1, figsize=(1075/100, 605/100), dpi=200)
fig.suptitle('Income Distribution', fontsize=18, fontweight='bold')

# Filter the dataframe to include only rows where 'AMT_INCOME_TOTAL' is less than or equal to 600,000
filtered_df = data1[data1['AMT_INCOME_TOTAL'] <= 600000]

# Create a histogram for the filtered income data
sns.histplot(filtered_df['AMT_INCOME_TOTAL'], ax=ax, bins=30)

# Set the title for the histogram
ax.set_title('Income Total ≤ 600,000', fontsize=16)

#plt.figure(figsize=(1075/100, 605/100))
plt.savefig('plot.png', dpi=200)
plt.show()

# %%
# Filter the dataframe to include only rows where 'AMT_INCOME_TOTAL' is less than or equal to 600,000
data1 = data1[data1['AMT_INCOME_TOTAL'] <= 600000]
data1.shape

# %%
data1.describe().T

# %%
# Create a subplot grid with rows equal to the number of continuous features and 2 columns
fig, axes = plt.subplots(len(conti_cols), 2, figsize=(1075/100, 605/100), dpi=200)

fig.suptitle('Box plots and Histograms for Continuous Features', fontsize=20, fontweight='bold')

# Loop through each continuous feature and its corresponding subplot row index
for i, variable in enumerate(conti_cols):
    # Create a box plot for the current feature in the left column
    sns.boxplot(x=data1[variable], ax=axes[i, 0])
    axes[i, 0].set_title(f'Box plot of {variable}', fontsize=16)
    
    # Create a histogram for the current feature in the right column, with KDE overlay
    sns.histplot(data1[variable], ax=axes[i, 1], kde=True)
    axes[i, 1].set_title(f'Histogram of {variable}', fontsize=16)

plt.tight_layout()
plt.savefig('plot2.png', dpi=200)
plt.show()


# %% [markdown]
# ### converting them to years and flipping them to positive values

# %%
# Convert 'DAYS_BIRTH' from negative days to positive years
data1['DAYS_BIRTH'] = abs(data1['DAYS_BIRTH']) / 365

# Convert 'DAYS_EMPLOYED' from negative days to positive years
data1['DAYS_EMPLOYED'] = abs(data1['DAYS_EMPLOYED']) / 365

# %%
# Create a subplot grid with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=200)

fig.suptitle('Box plot and Histogram for DAYS_BIRTH', fontsize=20, fontweight='bold')

# Create a box plot for 'DAYS_BIRTH' in the first subplot
sns.boxplot(x=data1['DAYS_BIRTH'], ax=axes[0])
axes[0].set_title('Box plot of DAYS_BIRTH', fontsize=16)

# Create a histogram with KDE for 'DAYS_BIRTH' in the second subplot
sns.histplot(data1['DAYS_BIRTH'], ax=axes[1], kde=True)
axes[1].set_title('Histogram of DAYS_BIRTH', fontsize=16)

# Display the final plot
plt.tight_layout()
plt.show()

# %%
# Determine the number of rows and columns of the subgraph
n_cols = 3
n_rows = (len(class_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi = 200)
fig.suptitle('Count Plots for Categorical Variables', fontsize=20, fontweight='bold')

# Flatten axes to facilitate indexing
axes = axes.flatten()

for i, col in enumerate(class_cols):
    sns.countplot(x=col, data=data1, ax=axes[i])
    axes[i].set_title(col, fontsize=16)

# Hide redundant subgraphs
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
for col in class_cols:
    plt.figure(figsize=(1075/100, 605/100), dpi=200)
    sns.countplot(x=col, data=data1)
    plt.title(col, fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{col}_count_plot.png')  # Save each plot as a separate file
    plt.close()  # Close the figure to free memory

# %%


# %% [markdown]
# There are missing data of the OCCUPATION_TYPE variable, so we remove missing data.

# %%
data1.info()

# %%
data1.dropna(inplace=True)
data1.shape

# %% [markdown]
# ## Feature Engineering

# %%
# Define a mapping from education levels to numeric values
education_mapping = {
    'Lower secondary': 1,
    'Secondary / secondary special': 2,
    'Incomplete higher': 3,
    'Higher education': 4,
    'Academic degree': 5
}

# Map the education levels in 'NAME_EDUCATION_TYPE' to numeric values using the defined mapping
data1['Education_Level'] = data1['NAME_EDUCATION_TYPE'].map(education_mapping)

# Drop the original 'NAME_EDUCATION_TYPE' column
data1.drop('NAME_EDUCATION_TYPE', axis=1, inplace=True)

class_cols.append('Education_Level')
class_cols.remove('NAME_EDUCATION_TYPE')

data1['Education_Level'].value_counts()

# %%
# Standardize and map 'CODE_GENDER' values: 'M' to 1, 'F' to 0
data1['CODE_GENDER'] = data1['CODE_GENDER'].str.strip().str.upper()
data1['CODE_GENDER'] = data1['CODE_GENDER'].map({'M': 1, 'F': 0})

# Standardize and map 'FLAG_OWN_REALTY' values: 'Y' to 1, 'N' to 0
data1['FLAG_OWN_REALTY'] = data1['FLAG_OWN_REALTY'].str.strip().str.upper()
data1['FLAG_OWN_REALTY'] = data1['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})

# Standardize and map 'FLAG_OWN_CAR' values: 'Y' to 1, 'N' to 0
data1['FLAG_OWN_CAR'] = data1['FLAG_OWN_CAR'].str.strip().str.upper()
data1['FLAG_OWN_CAR'] = data1['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})

print(data1['CODE_GENDER'].value_counts())
print(data1['FLAG_OWN_REALTY'].value_counts())
print(data1['FLAG_OWN_CAR'].value_counts())

# %%
# Define mappings for various categorical columns to numeric values
income_type_mapping = {
    'Student': 1,
    'Pensioner': 2,
    'State servant': 3,
    'Working': 4,
    'Commercial associate': 5
}

family_status_mapping = {
    'Widow': 1,
    'Separated': 2,
    'Single / not married': 3,
    'Civil marriage': 4,
    'Married': 5
}

housing_type_mapping = {
    'With parents': 1,
    'Rented apartment': 2,
    'Municipal apartment': 3,
    'Co-op apartment': 4,
    'Office apartment': 5,
    'House / apartment': 6
}

occupation_type_mapping = {
    'Low-skill Laborers': 1,
    'Cleaning staff': 2,
    'Cooking staff': 2,
    'Waiters/barmen staff': 2,
    'Security staff': 3,
    'Sales staff': 3,
    'Laborers': 3,
    'Drivers': 3,
    'Medicine staff': 4,
    'Secretaries': 4,
    'HR staff': 4,
    'Accountants': 5,
    'Core staff': 5,
    'Realty agents': 5,
    'Private service staff': 6,
    'High skill tech staff': 6,
    'Managers': 7,
    'IT staff': 7
}

# Map the categorical columns to their corresponding numeric values using the defined mappings
data1['NAME_INCOME_TYPE'] = data1['NAME_INCOME_TYPE'].map(income_type_mapping)
data1['NAME_FAMILY_STATUS'] = data1['NAME_FAMILY_STATUS'].map(family_status_mapping)
data1['NAME_HOUSING_TYPE'] = data1['NAME_HOUSING_TYPE'].map(housing_type_mapping)
data1['OCCUPATION_TYPE'] = data1['OCCUPATION_TYPE'].map(occupation_type_mapping)

# %%
# Calculate the number of rows needed based on the number of columns
n_cols = 3
n_rows = (len(class_cols) + n_cols - 1) // n_cols

# Create a subplot grid with the calculated number of rows and columns
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi=200)

fig.suptitle('Count Plots for Categorical Variables', fontsize=20, fontweight='bold')

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through each categorical column and create a count plot
for i, col in enumerate(class_cols):
    sns.countplot(x=col, data=data1, ax=axes[i])
    axes[i].set_title(col, fontsize=16)

# Remove any unused subplot axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust the layout to ensure subplots fit within the figure area without overlapping
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()

# %%
data1.describe().T

# %% [markdown]
# ## Credit Record EDA

# %% [markdown]
# - Month_Balance
#     - The month of the extracted data is the starting point, backwards, 0 is the current month, -1 is the previous month, and so on
# 
# - STATUS
#     - 0: 1-29 days past due 
#     - 1: 30-59 days past due 
#     - 2: 60-89 days overdue 
#     - 3: 90-119 days overdue 
#     - 4: 120-149 days overdue 
#     - 5: Overdue or bad debts, write-offs for more than 150 days 
#     - C: paid off that month 
#     - X: No loan for the month

# %%
data2.head()

# %%
# Define a mapping from status codes to point values
status_points = {'0': 2, '1': 0, '2': -2, '3': -5, '4': -10, '5': -20, 'C': 5, 'X': 3}

# Map the 'STATUS' column to point values using the defined mapping
data2['Points'] = data2['STATUS'].map(status_points)

# Group by 'ID' and sum the points for each group
scores_series = data2.groupby('ID')['Points'].sum()

# Reset the index of the series to create a DataFrame with 'ID' and 'Scores'
scores_df = scores_series.reset_index(name='Scores')

scores_df

# %%
# Define a function to evaluate credit ratings based on scores.
def credit_rating(score):
    if score > 100:
        return 'Good'
    elif score >= 0:
        return 'Average'
    else:
        return 'Bad'

# Apply this function to the 'Scores' column and create a new column 'Credit_Rating'
scores_df['Credit_Rating'] = scores_df['Scores'].apply(credit_rating)

# Drop the 'ID' column from the dataframe
scores_df.drop('Scores', axis=1, inplace=True)

scores_df

# %%
scores_df['Credit_Rating'].describe()

# %%
# Merge data1 with scores_df on the 'ID' column, using a left join
merge_df = pd.merge(data1, scores_df, on='ID', how='left')

# Drop rows where 'Scores' is NaN
merge_df = merge_df.dropna(subset=['Credit_Rating'])

merge_df.head(10)

# %%
# Remove useless column
merge_df.drop('FLAG_WORK_PHONE', axis=1, inplace=True)

# %% [markdown]
# now all the columns are numerical

# %%
merge_df.describe()

# %% [markdown]
# ## Classifier Model Building

# %% [markdown]
# Since there is no target variable in the application record, we will use the credit record to build a predictive model.

# %%
scaler = StandardScaler()
# Scale the continuous columns in merge_df
merge_df[conti_cols] = scaler.fit_transform(merge_df[conti_cols])

Credit_type_mapping = {
    'Good': 2,
    'Average': 1,
    'Bad': 0
}
# Code the 'Credit_Rating' column
merge_df['Credit_Rating'] = merge_df['Credit_Rating'].map(Credit_type_mapping)

# %%
# Drop the 'ID' column from the dataframe
# merge_df.drop('ID', axis=1, inplace=True)

# Define the feature matrix 'X' and the target variable 'y'
y = merge_df['Credit_Rating']
X = merge_df.drop('Credit_Rating', axis=1)


# Split the data into training and testing sets with 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train

# %% [markdown]
# ### CART Model

# %%
# Create a DecisionTreeClassifier (CART) model
cart_model = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
cart_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = cart_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a DataFrame to compare actual and predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Print the accuracy and the results DataFrame
print(f"Accuracy: {accuracy}")
print(results)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(class_report)

# %%
# Plot the confusion matrix
plt.figure(figsize=(1075/100, 605/100), dpi = 150)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=cart_model.classes_, yticklabels=cart_model.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('plot4.png', dpi=200)
plt.show()

# %% [markdown]
# ## Logistic Model

# %%
# Initialize the Logistic Regression model
log_model = LogisticRegression()

# Fit the model on the training data
log_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Create a DataFrame to compare actual and predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Print the accuracy and the results DataFrame
print(f"Accuracy: {accuracy}")
print(results)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1)

print("\nClassification Report:")
print(class_report)

# %%
# Plot the confusion matrix
plt.figure(figsize=(1075/100, 605/100), dpi = 150)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=log_model.classes_, yticklabels=log_model.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('plot5.png', dpi=200)
plt.show()

# %%



