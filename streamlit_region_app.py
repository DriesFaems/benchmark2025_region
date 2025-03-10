import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="European Scaleup Monitor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-weight: bold;
        color: #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('Aggregate region with country.xlsx')
    country = pd.read_excel('Aggregate country with country.xlsx')
    return df, country

df, country = load_data()

# Remove when Country ISO code is nan
df_long = df[df['Country'].notna()].reset_index(drop=True)

# Streamlit app header
st.markdown('<div class="main-header">European Scaleup Monitor: Regional Benchmarking Tool</div>', unsafe_allow_html=True)

# Add subheader
st.markdown(
    '<div class="sub-header">Compare growth metrics across different regions in Europe. '
    'For more information, visit <a href="https://scaleupinstitute.eu/" target="_blank">scaleupinstitute.eu</a> '
    'or contact <a href="mailto:info@scaleupinstitute.eu">info@scaleupinstitute.eu</a></div>', 
    unsafe_allow_html=True
)

# Create two columns for the filters
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Country selection
    countries = sorted(df_long['Country'].unique())
    selected_country = st.selectbox('Select country', countries, help="Choose a country to explore its regions")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Metrics mapping for better display
    metrics_mapping = {
        'Scaler': 'Scaler: Companies with 10%+ growth',
        'High Growth Firm': 'High Growth Firm: Companies with 20%+ growth',
        'Consistent High Growth Firm': 'Consistent High Growth Firm: Consistent 20%+ growth',
        'Consistent Hypergrower': 'Consistent Hypergrower: Consistent 40%+ growth',
        'Gazelle': 'Gazelle: Young high growth firms',
        'Mature High Growth Firm': 'Mature High Growth Firm: Mature high growth firms',
        'Scaleup': 'Scaleup: Young hypergrowers',
        'Superstar': 'Superstar: Mature hypergrowers'
    }
    
    selected_display = st.selectbox(
        'Select growth metric',
        list(metrics_mapping.values()),
        help="Choose which growth metric to visualize"
    )
    
    # Reverse mapping to get the internal metric name
    selected_user_metrics = list(metrics_mapping.keys())[list(metrics_mapping.values()).index(selected_display)]
    
    # Map to the actual column prefix in the dataset
    metrics_to_column = {
        'Scaler': 'Scaler',
        'High Growth Firm': 'HighGrowthFirm',
        'Consistent High Growth Firm': 'ConsistentHighGrowthFirm',
        'Consistent Hypergrower': 'VeryHighGrowthFirm',
        'Gazelle': 'Gazelle',
        'Mature High Growth Firm': 'Mature',
        'Scaleup': 'Scaleup',
        'Superstar': 'Superstar'
    }
    
    selected_metrics = metrics_to_column[selected_user_metrics]
    st.markdown('</div>', unsafe_allow_html=True)

# Filter data based on country selection
selection = df_long[df_long['Country'] == selected_country]
country_data = country[country['Country'] == selected_country]

# Region selection
regions = sorted(selection['Region in country'].unique())
selected_regions = st.multiselect('Select regions to compare', regions, help="Select multiple regions to benchmark against each other")

# Show meaningful content when no regions are selected
if not selected_regions:
    st.warning("👆 Please select at least one region to view the data")
    
    # Show a preview of available data
    st.markdown("### Preview of Available Data")
    st.dataframe(selection.head(5))
else:
    # Filtering data
    filtered_data = selection[selection['Region in country'].isin(selected_regions)]
    
    # Analysis Section
    st.markdown(f"### Analysis of {selected_display} in {selected_country}")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Trend Analysis", "📋 Detailed Data"])
    
    with tab1:
        # Enhanced visualization
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Set style
        sns.set_style("whitegrid")
        
        # Use a better color palette
        colors = sns.color_palette("tab10", len(selected_regions) + 1)  # +1 for country average
        
        # Years - updated to 2019-2023
        x = [2019, 2020, 2021, 2022, 2023]
        
        # First plot country average
        ylistcountry = []
        for year in x:
            col_name = f"{selected_metrics} {year} %"
            if col_name in country_data.columns:
                ylistcountry.append(country_data[col_name].iloc[0] * 100)
            else:
                ylistcountry.append(np.nan)  # Handle missing data
        
        ax.plot(x, ylistcountry, marker='o', linewidth=3, color=colors[0], label=f"{selected_country} (All regions)")
        
        # Add value labels for country average
        for j, val in enumerate(ylistcountry):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.2f}%", 
                    (x[j], val),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontweight='bold',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
                )
        
        # Then plot each region
        for i, region in enumerate(selected_regions):
            region_data = filtered_data[filtered_data['Region in country'] == region]
            ylist = []
            
            for year in x:
                col_name = f"{selected_metrics} {year} %"
                if col_name in region_data.columns:
                    ylist.append(region_data[col_name].iloc[0] * 100)
                else:
                    ylist.append(np.nan)  # Handle missing data
            
            ax.plot(x, ylist, marker='o', linewidth=2.5, color=colors[i+1], label=region)
            
            # Add value labels with better positioning
            for j, val in enumerate(ylist):
                if not np.isnan(val):
                    ax.annotate(
                        f"{val:.2f}%", 
                        (x[j], val),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        fontweight='bold',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
                    )
        
        # Better styling for the plot
        ax.set_title(f'Trend Analysis: {selected_display} (2019-2023)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Percentage of {selected_user_metrics}s (%)', fontsize=12, fontweight='bold')
        
        # Set x-axis to display only integers
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        
        # Add grid to the plot
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Improve legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(selected_regions) + 1), 
                 fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add region comparison for the latest year
        if len(selected_regions) > 1:
            st.markdown("### Regional Comparison (2023)")
            
            latest_data = []
            for region in selected_regions:
                region_data = filtered_data[filtered_data['Region in country'] == region]
                if f"{selected_metrics} 2023 %" in region_data.columns and f"{selected_metrics} 2023 Num" in region_data.columns:
                    latest_data.append({
                        'Region': region,
                        'Percentage': region_data[f"{selected_metrics} 2023 %"].iloc[0] * 100,
                        'Number': region_data[f"{selected_metrics} 2023 Num"].iloc[0],
                        'Total': region_data[f"{selected_metrics} 2023 Obs"].iloc[0]
                    })
            
            if latest_data:
                # Sort by percentage
                latest_data = sorted(latest_data, key=lambda x: x['Percentage'], reverse=True)
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(12, max(6, len(selected_regions) * 0.5)))
                
                regions_list = [item['Region'] for item in latest_data]
                percentages = [item['Percentage'] for item in latest_data]
                
                # Horizontal bar chart for better readability
                bars = ax.barh(regions_list, percentages, color=sns.color_palette("tab10", len(regions_list)))
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(
                        width + 0.3, 
                        bar.get_y() + bar.get_height()/2, 
                        f"{percentages[i]:.2f}%", 
                        ha='left', 
                        va='center',
                        fontweight='bold'
                    )
                
                # Add styling
                ax.set_title(f'Comparison of {selected_user_metrics}s in 2023', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel(f'Percentage of {selected_user_metrics}s (%)', fontsize=12, fontweight='bold')
                ax.set_xlim(0, max(percentages) * 1.2)  # Give some extra space for labels
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7, axis='x')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add additional information in columns
                st.markdown("#### Key Metrics (2023)")
                
                # Create columns for metrics
                cols = st.columns(len(latest_data))
                
                for i, item in enumerate(latest_data):
                    with cols[i]:
                        st.markdown(f"**{item['Region']}**")
                        st.metric(
                            label=f"{selected_user_metrics}s", 
                            value=f"{item['Number']:,}",
                            delta=None,
                            delta_color="off"
                        )
                        st.caption(f"Out of {item['Total']:,} total companies")
    
    with tab2:
        # Detailed data in a well-formatted table
        if not filtered_data.empty:
            # Get relevant columns
            listrelevantcolumns = ['Region in country']
            for col in filtered_data.columns:
                if selected_metrics in col:
                    listrelevantcolumns.append(col)
            
            # Create a new table with relevant columns
            newtable = filtered_data[listrelevantcolumns].copy()
            
            # Create a mapping dictionary for column renaming
            column_mapping = {}
            for col in newtable.columns:
                if col == 'Region in country':
                    continue
                
                if selected_metrics in col:
                    # Keep the original column name as the key
                    # Use the original column name as part of the new name to ensure uniqueness
                    parts = col.split(" ")
                    if len(parts) >= 3:
                        year = parts[1]
                        metric_type = parts[2]
                        
                        if metric_type == "Obs":
                            column_mapping[col] = f"Total companies ({year}) - {col}"
                        elif metric_type == "Num":
                            column_mapping[col] = f"Number of {selected_user_metrics}s ({year}) - {col}"
                        elif metric_type == "%":
                            column_mapping[col] = f"Percentage ({year}) - {col}"
            
            # Rename columns
            newtable = newtable.rename(columns=column_mapping)
            
            # Set index to region
            newtable = newtable.set_index('Region in country')
            
            # Format percentage columns
            for col in newtable.columns:
                if "Percentage" in col:
                    # First multiply by 100 to convert to percentage
                    newtable[col] = newtable[col] * 100
            
            # Display the table with improved styling and formatting
            st.dataframe(
                newtable.style.format({
                    col: "{:.2f}%" for col in newtable.columns if "Percentage" in col
                }),
                use_container_width=True,
                height=400
            )
            
            # Add download button
            # Create a copy for CSV export with formatted percentages
            csv_data = newtable.copy()
            for col in csv_data.columns:
                if "Percentage" in col:
                    csv_data[col] = csv_data[col].round(2).astype(str) + '%'
            
            csv = csv_data.to_csv()
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"{selected_country}_{selected_metrics}_regional_data.csv",
                mime="text/csv",
            )
