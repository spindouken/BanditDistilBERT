#!/usr/bin/env python3

import pandas as pd
import re

# load CSV file into a pandas DataFrame
df = pd.read_csv('/home/psychesophy/New folder/job_descriptions.csv')

# skills we're looking for
skills = [
    'Python',
    'JavaScript',
    'C',
    'R',
    'TypeScript',
    'C#',
    'HTML',
    'CSS',
    'Bash Scripting',
    'React',
    'Flask',
    'Express.js',
    '.NET SDK',
    'SQLAlchemy',
    'Sequelize',
    'MySQL',
    'Node.js',
    'Docker',
    'Linux',
    'Nginx',
    'GCP (Google Cloud Platform)',
    'Bootstrap',
    'jQuery',
    'Pandas',
    'NumPy',
    'Matplotlib',
    'Seaborn',
    'PowerBI',
    'TensorFlow',
    'PyTorch',
    'scikit-learn',
    'Keras',
    'Hugging Face Transformers',
    'XGBoost',
    'LightGBM',
    'CatBoost',
    'OpenCV',
    'Biopython',
    'fMRI Analysis Tools',
    'Figma',
    'Git',
    'GitHub',
    'Machine Learning',
    'Data Science',
    'Chemical Engineering',
    'Mathematics',
    'API'
]

# check if any of the skills are in the combined text
def contains_skills(description, listed_skills, skills):
    combined_text = f"{description} {listed_skills}"
    combined_text_lower = combined_text.lower()  # case-insensitive matching
    return any(skill.lower() in combined_text_lower for skill in skills), combined_text

# extract labels from the combined text
def extract_labels(text, skills):
    labels = []
    text_lower = text.lower()  # case-insensitive matching
    for skill in skills:
        if skill == 'C':
            # regex to capture "C" in various contexts:
            # - surrounded by spaces
            # - at the start or end of a string
            # - preceded by a comma
            # - enclosed in parentheses
            # - surrounded by any non-word characters
            if re.search(r'(?<=\s)C(?=\s)|^C(?=\s)|(?<=\s)C$|,\s*C(?=\s)|\(C(?=\s)|(?<=\()\s*C\s*(?=\))|(?<=\W)C(?=\W)', text):
                labels.append(skill)
        elif skill == 'R':
            # regex to capture "R" in various contexts (see C comments)
            if re.search(r'(?<=\s)R(?=\s)|^R(?=\s)|(?<=\s)R$|,\s*R(?=\s)|\(R(?=\s)|(?<=\()\s*R\s*(?=\))|(?<=\W)R(?=\W)', text):
                labels.append(skill)
        else:
            if skill.lower() in text_lower:
                labels.append(skill)
    return labels


# apply functions to filter dataframe and concatenated text
df['contains_skill'], df['Combined_Description'] = zip(*df.apply(lambda row: contains_skills(row['Job Description'] if pd.notnull(row['Job Description']) else '', 
                                                                                             row['skills'] if pd.notnull(row['skills']) else '', 
                                                                                             skills), axis=1))

# filter datafram to keep only rows that contain the skills
filtered_df = df[df['contains_skill']]

# extract labels
filtered_df['Labels'] = filtered_df['Combined_Description'].apply(lambda x: extract_labels(x, skills))

# keep only rows that have at least one label
filtered_df = filtered_df[filtered_df['Labels'].str.len() > 0]

# keep only the necessary columns: combined description and labels
filtered_df = filtered_df[['Combined_Description', 'Labels']]

# save filtered and processed dataframe
filtered_df.to_csv('final_filtered!.csv', index=True)

print(f"Total number of unfiltered job postings: {len(df)}")

# print number of matching rows
print(f"Number of matching job postings: {len(filtered_df)}")
