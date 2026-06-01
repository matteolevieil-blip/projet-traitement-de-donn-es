# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt
import seaborn as sns
from math import pi


df = pd.read_csv('/Users/matteo/Documents/Polytech/MAM3/Projets/base_donnees/skill_requirement.csv')
df.head()
df.info()


df.isnull().sum()

df = df.drop_duplicates()

# Sort the data by Total score in descending order
df = df.sort_values(by='Total', ascending=False)

# Set the theme for the plot
sns.set_theme(style="whitegrid")

# Create a bar plot of the top 10 sports
plt.figure(figsize=(14, 8))
ax = sns.barplot(x='SPORT', y='Total', data=df.head(10), palette="viridis")

# Set plot title and labels with larger font sizes for better readability
ax.set_title('Top 10 Sports by Total Score', fontsize=20, weight='bold')
ax.set_xlabel('Sport', fontsize=16, weight='bold')
ax.set_ylabel('Total Score', fontsize=16, weight='bold')

# Rotate x-tick labels for better readability
plt.xticks(rotation=45, fontsize=12, ha='right')

# Add annotations to each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9),
                textcoords='offset points', fontsize=12, weight='bold')

# Display the plot
plt.tight_layout()
plt.show()




# Create a correlation matrix
corr = df[['Endurance', 'Strength', 'Power', 'Speed', 'Agility', 'Flexibility', 'Nerve', 'Durability', 'Hand-eye coordination', 'Analytical Aptitude']].corr()

# Create a heatmap
plt.figure(figsize=(11, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Between Skill Requirements', fontsize=16)
plt.show()














