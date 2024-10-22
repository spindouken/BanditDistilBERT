#!/usr/bin/env python3

import pandas as pd
import re

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('/home/psychesophy/New folder/job_descriptions.csv')

# Select columns to retain for comparison
columns_to_keep = [
    'Job Title', 
    'Company', 
    'Salary Range', 
    'location', 
    'Experience', 
    'Job Description', 
    'skills'
]

# Create a new dataframe with only the selected columns
df_filtered = df[columns_to_keep]

# Define the set of words or skills you're looking for
skills = [
    'Python', 'JavaScript', 'C', 'R', 'TypeScript', 'C#', 'HTML', 'CSS',
    'Bash Scripting', 'React', 'Flask', 'Express.js', '.NET SDK', 'SQLAlchemy',
    'Sequelize', 'MySQL', 'Node.js', 'Docker', 'Linux', 'Nginx',
    'GCP (Google Cloud Platform)', 'Bootstrap', 'jQuery', 'Pandas', 'NumPy',
    'Matplotlib', 'Seaborn', 'PowerBI', 'TensorFlow', 'PyTorch',
    'scikit-learn', 'Keras', 'Hugging Face Transformers', 'XGBoost', 'LightGBM',
    'CatBoost', 'OpenCV', 'Biopython', 'fMRI Analysis Tools', 'Figma', 'Git',
    'GitHub', 'Machine Learning', 'Data Science', 'API'
]

# Function to check if any of the skills are in the combined text
def contains_skills(description, listed_skills, skills):
    combined_text = f"{description} {listed_skills}"
    combined_text_lower = combined_text.lower()  # Ensure case-insensitive matching
    return any(skill.lower() in combined_text_lower for skill in skills), combined_text

# Apply the function to filter the DataFrame and concatenate the text
df_filtered['contains_skill'], df_filtered['Combined_Description'] = zip(*df_filtered.apply(
    lambda row: contains_skills(
        row['Job Description'] if pd.notnull(row['Job Description']) else '', 
        row['skills'] if pd.notnull(row['skills']) else '', 
        skills), 
    axis=1))

# Filter the DataFrame to keep only rows that contain the desired skills
filtered_df = df_filtered[df_filtered['contains_skill']]

# Export filtered dataset to CSV for Power BI import
filtered_df.to_csv('filtered_job_postings.csv', index=False)
