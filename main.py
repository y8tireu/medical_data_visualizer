import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (df['BMI'] > 25).astype(int)
df.drop('BMI', axis=1, inplace=True)

# Normalize cholesterol and gluc
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Function to draw categorical plot
def draw_cat_plot():
    # Melt DataFrame
    df_cat = pd.melt(df, id_vars=["cardio"], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Draw the catplot
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', 
                    data=df_cat, kind='bar', height=5, aspect=1)

    # Get the figure for the output
    fig = g.fig

    # Return the figure
    return fig

# Function to draw heat map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', center=0, ax=ax)

    # Return the figure
    return fig

# Main execution block
if __name__ == "__main__":
    # Draw and save categorical plot
    cat_fig = draw_cat_plot()
    cat_fig.savefig('catplot.png')

    # Draw and save heat map
    heatmap_fig = draw_heat_map()
    heatmap_fig.savefig('heatmap.png')
