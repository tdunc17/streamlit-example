import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

# Set page title and favicon.
st.set_page_config(
    page_title="Keg-O-Rater", 
)


########## Main Panel

# Header and Description
st.write("""
# Keg-O-Rater
Description
""")

##########


######### Sidebar
st.sidebar.header('Keg-O-Rater')
st.sidebar.caption("Team Gemini - Erdos Institute - Spring 2022 Cohort")

st.sidebar.markdown("description")
# Social Links
st.sidebar.write("""
more words
"""
)
st.sidebar.markdown("----")

st.sidebar.header('Description')
st.sidebar.markdown("""
Recommendations are generated from a list of ____ unique beers from ____ different breweries and take into consideration the following aspects of each beer:
* **Alcohol content** (% by volume)
* **Minimum and maximum IBU** (International Bitterness Units)
* **Mouthfeel**
   * Astringency
   * Body
   * Alcohol
* **Taste**
   * Bitter
   * Sweet
   * Sour
   * Salty
* **Flavor And Aroma**
   * Fruity
   * Hoppy
   * Spices
   * Malty
""")
st.sidebar.markdown("----")


# Data Preprocessing

full_data = pd.read_csv('updated_beer_profile_and_ratings.csv')

# List numeric features (columns) of different types
tasting_profile_cols = ['Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']
chem_cols = ['ABV', 'Min IBU', 'Max IBU']

# Scaling data
def scale_col_by_row(df, cols):
    scaler = MinMaxScaler()
    # Scale values by row
    scaled_cols = pd.DataFrame(scaler.fit_transform(df[cols].T).T, columns=cols)
    df[cols] = scaled_cols
    return df

def scale_col_by_col(df, cols):
    scaler = MinMaxScaler()
    # Scale values by column
    scaled_cols = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)
    df[cols] = scaled_cols
    return df

# Scale values in tasting profile features (across rows)
data = scale_col_by_row(full_data, tasting_profile_cols)

# Scale values in tasting profile features (across columns)
data = scale_col_by_col(full_data, tasting_profile_cols)

# Scale values in chemical features (across columns)
data = scale_col_by_col(full_data, chem_cols)


# Use only numeric features for determining nearest neighbors
df_num = data.select_dtypes(exclude=['object'])


########## Main Panel

# Setting input parameters (collecting user input)

st.markdown("----")
st.markdown("\n")

# User Input

def user_input_features():
    Style = st.selectbox("What's your favorite beer style?", (data['Style'].unique()))

    style_string = "Which " + Style + " have you enjoyed recently?"
    Beer = st.selectbox(style_string, sorted(data[data['Style'] == Style]['Beer Name (Full)'].unique()))

    user_input = Beer

    # Locate numerical features of user inputted beer
    test_data = data[data["Beer Name (Full)"] == user_input]
    num_input = df_num.loc[test_data.index].values
    
    # Detect beer style
    style_input = test_data['Style'].iloc[0]

    return num_input, style_input

num_input, style_input = user_input_features()

##########


# Generate recommendations based on user input
def get_neighbors(data, num_input, style_input, same_style=False):
    if same_style==True:
        # Locate beers of same style
        df_target = data[data["Style"] == style_input].reset_index(drop=True)
    else:
        # Locate beers of different styles
        df_target = data[data["Style"] != style_input].reset_index(drop=True)

    df_target_num = df_num.loc[df_target.index]
    # Calculate similarities
    search = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(df_target_num)
    _ , queried_indices = search.kneighbors(num_input)
    # Top 5 recommendations
    target_rec_df = df_target.loc[queried_indices[0][1:]]
    target_rec_df = target_rec_df.sort_values(by=['review_overall'], ascending=False)
    target_rec_df = target_rec_df[['Name', 'Brewery', 'Style', 'review_overall']]
    target_rec_df.index = range(1, 6)
    target_rec_df.drop('review_overall', axis=1, inplace=True)
    return target_rec_df


########## Main Panel
st.markdown("\n")
st.markdown("\n")

# Add button to generate recommendations
st.write("*Ready to check out your recommendations?*")
display_recommendation_now = st.button('Keg me!')
if display_recommendation_now:
    # Generate recommendations
    st.header('Recommended Beers:')

    # List recommended beers with the same style
    st.subheader('Tried and True Style')
    top_5_same_style_rec = get_neighbors(data, num_input, style_input, same_style=True)
    st.dataframe(top_5_same_style_rec)


    # List recommended beers with different styles
    st.subheader('Suprise Me')
    top_5_diff_style_rec = get_neighbors(data, num_input, style_input, same_style=False)
    st.dataframe(top_5_diff_style_rec)

##########