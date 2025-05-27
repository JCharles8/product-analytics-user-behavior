#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Funnel Analysis of Simulated Productivity Tool Users

# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt

# --- Load Data ---
df = pd.read_csv("Simulated_Productivity_Tool_Usage_Data.csv")
df['action_date'] = pd.to_datetime(df['action_date'])
df['signup_date'] = pd.to_datetime(df['signup_date'])

# --- Preview Data ---
df.head()

# --- Define Funnel Steps ---
funnel_steps = ['login', 'create_page', 'use_template', 'invite_collab']

# --- Step 1: Identify User Completion at Each Step ---
user_funnels = {}

for step in funnel_steps:
    users_with_action = df[df['action'] == step]['user_id'].unique()
    user_funnels[step] = set(users_with_action)

# --- Step 2: Calculate Funnel Counts ---
step_counts = []
users_remaining = user_funnels[funnel_steps[0]]

for step in funnel_steps:
    users_remaining = users_remaining & user_funnels[step] if step != funnel_steps[0] else user_funnels[step]
    step_counts.append((step, len(users_remaining)))

# --- Step 3: Convert to DataFrame ---
funnel_df = pd.DataFrame(step_counts, columns=['step', 'users'])
funnel_df['conversion_rate'] = funnel_df['users'] / funnel_df['users'].iloc[0]

# --- Step 4: Plot Funnel ---
plt.figure(figsize=(8, 5))
plt.bar(funnel_df['step'], funnel_df['users'], color='skyblue')
plt.title("User Funnel Analysis")
plt.ylabel("Number of Users")
plt.xlabel("Funnel Step")
plt.grid(axis='y')
plt.show()

# --- Step 5: Display Final Funnel DataFrame ---
funnel_df


# In[3]:


# Retention Analysis: Tracking Returning Users Over Time

# Step 1: Define a user's "active day" as any day they performed an action
df['days_since_signup'] = (df['action_date'] - df['signup_date']).dt.days

# Step 2: Create retention matrix
retention_matrix = (
    df.groupby(['user_id', 'days_since_signup'])
    .size()
    .reset_index(name='activity')
    .pivot(index='user_id', columns='days_since_signup', values='activity')
    .fillna(0)
)

# Convert to binary (1 = active, 0 = inactive)
retention_matrix = retention_matrix.applymap(lambda x: 1 if x > 0 else 0)

# Step 3: Calculate retention percentages at specific days
retention_days = [1, 7, 14, 30, 60]
retention_curve = retention_matrix[retention_days].mean().reset_index()
retention_curve.columns = ['day', 'retention_rate']

# Step 4: Plot retention curve
plt.figure(figsize=(8, 5))
plt.plot(retention_curve['day'], retention_curve['retention_rate'], marker='o', color='mediumslateblue')
plt.title("User Retention Over Time")
plt.xlabel("Days Since Signup")
plt.ylabel("Retention Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.xticks(retention_days)
plt.show()

# Step 5: Display retention table
retention_curve


# In[4]:


# 📘 Power User Segmentation: Who Are the Most Engaged Users?

# Step 1: Total actions per user
user_action_counts = df.groupby('user_id').size().reset_index(name='total_actions')

# Step 2: Power user threshold (e.g., top 20% most active)
threshold = user_action_counts['total_actions'].quantile(0.80)
power_users = user_action_counts[user_action_counts['total_actions'] >= threshold]
regular_users = user_action_counts[user_action_counts['total_actions'] < threshold]

# Step 3: Merge back to main df
df['user_type'] = df['user_id'].apply(lambda x: 'power' if x in set(power_users['user_id']) else 'regular')

# Step 4: Compare action distribution by user type
action_dist = df.groupby(['user_type', 'action']).size().reset_index(name='count')
action_dist_pivot = action_dist.pivot(index='action', columns='user_type', values='count').fillna(0)

# Step 5: Normalize counts by total users in each group
power_user_count = len(power_users)
regular_user_count = len(regular_users)
action_dist_pivot['power_pct'] = action_dist_pivot['power'] / power_user_count
action_dist_pivot['regular_pct'] = action_dist_pivot['regular'] / regular_user_count

# Step 6: Plot comparison
action_dist_pivot[['power_pct', 'regular_pct']].plot(kind='bar', figsize=(10, 5))
plt.title("Feature Usage Comparison: Power vs. Regular Users")
plt.ylabel("Avg. Usage per User")
plt.xlabel("Action")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Optional: View Data
action_dist_pivot[['power_pct', 'regular_pct']]

