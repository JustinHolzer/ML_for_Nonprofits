import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of donors to simulate
num_donors = 1000

# Simulate donor-specific characteristics
ages = np.random.randint(20, 80, size=num_donors)
genders = np.random.choice(['Male', 'Female'], size=num_donors)
income_levels = np.random.choice(['Low', 'Medium', 'High'], size=num_donors, p=[0.3, 0.5, 0.2])
education_levels = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=num_donors, p=[0.3, 0.4, 0.2, 0.1])
occupations = np.random.choice(['Employed', 'Self-Employed', 'Retired', 'Student', 'Unemployed'], size=num_donors)

# Simulate donation-specific characteristics
last_donation_amounts = np.round(np.random.exponential(scale=50, size=num_donors), 2)
total_donations = np.round(np.random.exponential(scale=500, size=num_donors), 2)
num_donations = np.random.poisson(lam=5, size=num_donors)
recency = np.random.randint(1, 365, size=num_donors)  # days since last donation

# Simulate engagement and relationship metrics
email_open_rates = np.random.uniform(0, 1, size=num_donors)
event_participation = np.random.choice(['Yes', 'No'], size=num_donors, p=[0.2, 0.8])
social_media_engagement = np.random.choice(['High', 'Medium', 'Low'], size=num_donors, p=[0.1, 0.4, 0.5])

# Simulate target variable: whether the donor will donate again (1: Yes, 0: No)
donate_again = np.random.choice([0, 1], size=num_donors, p=[0.7, 0.3])

# Create a DataFrame
donor_data = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'Income Level': income_levels,
    'Education Level': education_levels,
    'Occupation': occupations,
    'Last Donation Amount': last_donation_amounts,
    'Total Donations': total_donations,
    'Number of Donations': num_donations,
    'Recency (days)': recency,
    'Email Open Rate': email_open_rates,
    'Event Participation': event_participation,
    'Social Media Engagement': social_media_engagement,
    'Donate Again': donate_again
})

# Display the first few rows of the DataFrame
print(donor_data.head())

# Save to CSV for further use
donor_data.to_csv('donor_prediction_data.csv', index=False)
