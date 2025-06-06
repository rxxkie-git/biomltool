#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import ipywidgets as widgets
from IPython.display import display, clear_output


# In[2]:


# Set seed and create dummy data
np.random.seed(42)
years = list(range(2000, 2026))
animals = ['Deer', 'Fox', 'Rabbit', 'Bear', 'Wolf', 'Tiger', 'Elephant', 'Zebra', 'Lion', 'Giraffe']

data = []

for animal in animals:
    for year in years:
        birth_rate = np.random.randint(20, 100)
        death_rate = np.random.randint(10, 90)
        birth_death_ratio = round(birth_rate / (death_rate + 1e-3), 2)

        # Simulate status dynamically from ratio trend
        endangered = birth_death_ratio < 1.0

        data.append({
            'species': animal,
            'year': year,
            'birth_rate': birth_rate,
            'death_rate': death_rate,
            'birth_death_ratio': birth_death_ratio,
            'endangered': endangered
        })

df = pd.DataFrame(data)


# In[3]:


# Encode categorical values
le = LabelEncoder()
df['species_code'] = le.fit_transform(df['species'])

# Define features and target
features = ['species_code', 'year', 'birth_rate', 'death_rate', 'birth_death_ratio']
X = df[features]
y = df['endangered']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Mapping for decoding species
df_encoded = df[['species', 'species_code']].drop_duplicates()

years_plot = pd.date_range(start='2000', end='2025', freq='Y')
n = len(years_plot)

np.random.seed(1)
trend_data = {
    animal: np.linspace(100, np.random.randint(70, 95), n) + np.random.normal(0, 2, n)
    for animal in animals
}

df_animals = pd.DataFrame(trend_data, index=years_plot)


# In[4]:


# Dropdowns for user selection
species_dropdown = widgets.Dropdown(
    options=animals,
    description='Species:',
    style={'description_width': 'initial'}
)

year_dropdown = widgets.Dropdown(
    options=years,
    description='Year:',
    style={'description_width': 'initial'}
)

# Predict button
predict_button = widgets.Button(
    description='Predict Status',
    button_style='info'
)

# Output area for results and plots
output = widgets.Output()


# In[5]:


# Function to run on button click
def on_predict_clicked(b):
    with output:
        clear_output()
        species = species_dropdown.value
        year = year_dropdown.value

        # Filter the data for the selected animal and year
        filtered = df[(df['species'] == species) & (df['year'] == year)]

        if not filtered.empty:
            birth = int(filtered['birth_rate'].values[0])
            death = int(filtered['death_rate'].values[0])
            ratio = round(filtered['birth_death_ratio'].values[0], 2)

            input_data = filtered[['birth_rate', 'death_rate', 'birth_death_ratio']].copy()
            input_data['species_code'] = df_encoded[df_encoded['species'] == species]['species_code'].iloc[0]
            input_data['year'] = year
            input_data = input_data[['species_code', 'year', 'birth_rate', 'death_rate', 'birth_death_ratio']]

            prediction = model.predict(input_data)[0]
            status = "🛑 <span style='color:red'><b>Endangered</b></span>" if prediction else "✅ <span style='color:green'><b>Not Endangered</b></span>"

            display(widgets.HTML(f"""
                <h4>📊 Prediction Summary</h4>
                <ul>
                    <li><b>Species:</b> {species}</li>
                    <li><b>Year:</b> {year}</li>
                    <li><b>Birth Rate:</b> {birth}</li>
                    <li><b>Death Rate:</b> {death}</li>
                    <li><b>Birth/Death Ratio:</b> {ratio}</li>
                    <li><b>Status:</b> {status}</li>
                </ul>
            """))

            # --- Extinction Prediction Logic ---
            species_code_val = df_encoded[df_encoded['species'] == species]['species_code'].iloc[0]
            
            # Simulate future data for extinction prediction
            extinction_year = None
            future_years = range(year + 1, year + 50) # Predict 50 years into the future

            # Get the last known birth and death rates
            last_birth_rate = birth
            last_death_rate = death

            for future_year in future_years:
                # Simple linear decay for birth rate and increase for death rate
                # You might want a more sophisticated model here
                future_birth_rate = max(1, last_birth_rate * (1 - (future_year - year) * 0.02)) # 2% decline per year
                future_death_rate = min(99, last_death_rate * (1 + (future_year - year) * 0.01)) # 1% increase per year
                
                if future_death_rate <= 0: # Avoid division by zero
                    future_birth_death_ratio = float('inf') 
                else:
                    future_birth_death_ratio = round(future_birth_rate / future_death_rate, 2)


                future_input = pd.DataFrame([[species_code_val, future_year, future_birth_rate, future_death_rate, future_birth_death_ratio]],
                                            columns=features)
                
                future_prediction = model.predict(future_input)[0]

                # If predicted as endangered and birth_death_ratio is very low (e.g., <= 0.5), consider it extinct
                if future_prediction and future_birth_death_ratio <= 0.5:
                    extinction_year = future_year
                    break
            
            if extinction_year:
                display(widgets.HTML(f"""
                    <h4>📉 Extinction Forecast</h4>
                    <p>Based on current trends and model prediction, {species} could become extinct around the year <b>{extinction_year}</b> if conditions worsen and the birth/death ratio continues to decline below 0.5 when endangered.</p>
                """))
            else:
                display(widgets.HTML(f"""
                    <h4>📈 Extinction Forecast</h4>
                    <p>{species} is not predicted to go extinct within the next 50 years under current declining trend assumptions, or if its birth/death ratio doesn't consistently fall below 0.5 when endangered.</p>
                """))
            # --- End Extinction Prediction Logic ---


            # Plot population trend with annotations
            if species in df_animals.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_animals.index, df_animals[species], label=species, color='tab:blue', linewidth=2)
                ax.axvline(pd.Timestamp(f'{year}-01-01'), color='red', linestyle='--', label='Selected Year')

                # Shaded periods with labels
                recession_periods = [
                    ('2008-01-01', '2009-12-31', 'Global Financial Crisis'),
                    ('2020-01-01', '2021-12-31', 'COVID-19 Pandemic')
                ]

                for start, end, label in recession_periods:
                    start_dt = pd.to_datetime(start)
                    end_dt = pd.to_datetime(end)
                    mid_point = start_dt + (end_dt - start_dt) / 2

                    ax.axvspan(start_dt, end_dt, color='gray', alpha=0.3)
                    ax.text(mid_point, ax.get_ylim()[1]*0.95, label,
                            horizontalalignment='center', verticalalignment='top',
                            fontsize=9, color='black', rotation=90,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

                ax.set_title(f"{species} Population Trend (2000–2025)", fontsize=14)
                ax.set_ylabel("Population Index")
                ax.set_xlabel("Year")
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.show()
        else:
            print("❌ Data not found for the selected inputs.")


# In[6]:


# Link function to button
predict_button.on_click(on_predict_clicked)


# In[7]:


# Display widgets on screen
display(widgets.VBox([species_dropdown, year_dropdown, predict_button, output]))


# In[ ]:




