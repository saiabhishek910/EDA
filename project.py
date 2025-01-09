import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Function for normalization (min-max scaling)
def normalize_data(df, selected_cols):
    # Check if selected columns contain numeric data
    numeric_cols = df[selected_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.success(f"Normalized columns: {', '.join(numeric_cols)}")
    else:
        st.error("Selected columns do not contain numeric data for normalization.")
    return df

# Function for standardization (Z-score)
def standardize_data(df, selected_cols):
    # Check if selected columns contain numeric data
    numeric_cols = df[selected_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.success(f"Standardized columns: {', '.join(numeric_cols)}")
    else:
        st.error("Selected columns do not contain numeric data for standardization.")
    return df

# Function for log transformation
def log_transform(df, selected_col):
    if df[selected_col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        df[selected_col] = np.log1p(df[selected_col])
        st.success(f"Log-transformed column: {selected_col}")
    else:
        st.error("Selected column is not numeric and cannot be log-transformed.")
    return df

# Function for label encoding
def label_encode(df, selected_col):
    encoder = LabelEncoder()
    df[selected_col] = encoder.fit_transform(df[selected_col])
    return df

def convert_to_datetime(df, column_name):
    try:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        st.success(f"Successfully converted column '{column_name}' to datetime.")
        # Display rows with invalid datetime entries
        invalid_rows = df[df[column_name].isna()]
        if not invalid_rows.empty:
            st.warning(f"Some values in column '{column_name}' could not be converted to datetime and were set to NaT.")
            st.write("Invalid rows:")
            st.write(invalid_rows)
    except Exception as e:
        st.error(f"Error while converting column '{column_name}' to datetime: {e}")
    return df

def data_transformation(df):
    st.header('Data Transformation')

    if st.checkbox("Normalize Data"):
        selected_cols = st.multiselect("Select Columns for Normalization", df.columns)
        if selected_cols:
            df = normalize_data(df, selected_cols)
            st.write(df.head())

    if st.checkbox("Standardize Data"):
        selected_cols = st.multiselect("Select Columns for Standardization", df.columns)
        if selected_cols:
            df = standardize_data(df, selected_cols)
            st.write(df.head())

    if st.checkbox("Log Transformation"):
        selected_col = st.selectbox("Select Column for Log Transformation", df.select_dtypes(include=[np.number]).columns)
        if selected_col:
            df = log_transform(df, selected_col)
            st.write(df.head())

    if st.checkbox("Label Encoding"):
        selected_col = st.selectbox("Select Column for Label Encoding", df.select_dtypes(include=[object]).columns)
        if selected_col:
            df = label_encode(df, selected_col)
            st.write(df.head())

    if st.checkbox("Convert to Datetime and Extract Features"):
        selected_col = st.selectbox("Select Column for Date Conversion", df.select_dtypes(include=[object]).columns)
        if selected_col:
            df = convert_to_datetime(df, selected_col)
            st.write(df.head())

    return df


# main function
def main():
    # Sidebar
    categories = ['EDA', 'Plots','Data Transformation']
    t = st.sidebar.selectbox("Select Your Category", categories)
    st.title("Interactive Data Analysis and Transformation Tool Using Streamlit")
    # Upload dataset
    data = st.file_uploader("Please Upload Your Dataset", type=['csv', 'xlsx', 'txt'])
    if data is not None:
        if data.name.endswith('.csv'):
            df = pd.read_csv(data)
        elif data.name.endswith('.txt'):
            df = pd.read_csv(data)
        elif data.name.endswith('.xlsx'):
            df = pd.read_excel(data)
        else:
            st.error("Unsupported File Format")
            return
        # Store the dataframe in session state
        st.session_state.df = df

    # Check if data is in session state
    if 'df' in st.session_state:
        df = st.session_state.df

        if t == 'EDA':
            st.header('Exploratory Data Analysis')
            if st.checkbox("Show Dataset"):
                num_rows = st.slider("Select Number of Rows to Display", 5, len(df), 5)
                st.write(df.head(num_rows)) 
            if st.checkbox("Show Dataset Shape"):
                st.write(df.shape)
            if st.checkbox("Describe Dataset"):
                st.write(df.describe())
            if st.checkbox("Dataset Columns"):
                st.write(df.columns)
            if st.checkbox("Show Selected Columns Data"):
                selected = st.multiselect("Select Column(s)", df.columns)
                if selected:
                    st.write(df[selected])
                else:
                    st.error("No Columns Selected")
            if st.checkbox("Show Value Counts"):
                selected_coloumns = st.selectbox("Select Column", df.columns)
                if selected_coloumns:
                    st.write(df[selected_coloumns].value_counts())
                else:
                    st.write("No Columns Selected")
            
            if st.checkbox("Show Correlation Matrix"):
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    st.error("No Numeric Columns Available For Correlation Analysis.")
                else:
                    corr_matrix = numeric_df.corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
            if st.checkbox("Check for Missing Values"):
                missing_values = df.isnull().sum()
                if missing_values.sum() == 0:
                    st.success("This dataset has no missing values!")
                else:
                    st.write(missing_values)
                    st.bar_chart(missing_values)
            if st.checkbox("Check for Duplicates"):
                duplicates = df.duplicated().sum()
                if duplicates==0:
                    st.success("This dataset has no duplicate values!")
                else:
                    st.write(f"Number Of Duplicates: {duplicates}")
                    if duplicates > 0 and st.button("Drop Duplicates"):
                        df = df.drop_duplicates()
                        st.write("Duplicates Removed")
            if st.checkbox("Show Data Types"):
                st.write(df.dtypes)
            if st.checkbox("Show Outlier Detection"):
                selected_col = st.selectbox("Select Column for Box Plot", df.select_dtypes(include=np.number).columns)
                if selected_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(y=df[selected_col], ax=ax)
                    ax.set_title(f"Box Plot for {selected_col}")
                    st.pyplot(fig)
                else:
                    st.error("No Numeric Columns Available For Outlier Detection")
            if st.checkbox("Show Pairplot"):
                selected_cols = st.multiselect("Select Columns for Pairplot", df.select_dtypes(include=np.number).columns)
                if len(selected_cols) > 1:
                    selected_df = df[selected_cols]
                    fig = plt.figure(figsize=(10, 8))
                    sns.pairplot(selected_df)
                    st.pyplot(fig)
                else:
                    st.error("No Numeric Columns Available To Show Pairplot")
            if st.checkbox("Show Distribution Plots"):
                selected_col = st.selectbox("Select Column for Distribution Plot", df.select_dtypes(include=np.number).columns)
                if selected_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[selected_col], kde=True, ax=ax)
                    st.pyplot(fig)
                else:
                    st.error("No Numeric Columns Available for Distribution Plots")

        elif t == 'Plots':
            st.header('Plots')
            if st.checkbox("Show Bar Chart"):
                selected_col = st.selectbox("Select Column for Bar Chart", df.select_dtypes(include='object').columns)
                if selected_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[selected_col].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(f"Bar Chart of {selected_col}")
                    st.pyplot(fig)
                else:
                    st.error("This dataset doesn't contain any valid categorical columns for the Bar Chart.")

            if st.checkbox("Show Pie Chart"):
                selected_col = st.selectbox("Select Column for Pie Chart", df.select_dtypes(include='object').columns)   
                if selected_col:  
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[selected_col].value_counts().plot(kind='pie', ax=ax)
                    ax.set_title(f"Pie Chart of {selected_col}")
                    st.pyplot(fig)
                else:
                    st.error("This dataset doesn't contain any valid categorical columns for the Pie Chart.")
            if st.checkbox("Show Line Plot"):
                selected_col = st.selectbox("Select Column for Line Plot (Y-axis)", df.select_dtypes(include=np.number).columns)
                if selected_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(df.index, df[selected_col])
                    ax.set_title(f"Line Plot for {selected_col}")
                    ax.set_xlabel("Index")
                    ax.set_ylabel(selected_col)
                    st.pyplot(fig)
                else:
                    st.error("This dataset doesn't contain any valid numerical columns for the Line Plot.")
            if st.checkbox("Show Area Chart"):
                selected_col = st.selectbox("Select Column for Area Chart", df.select_dtypes(include=np.number).columns)
                if selected_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.fill_between(df.index, df[selected_col], color="skyblue", alpha=0.4)
                    ax.plot(df.index, df[selected_col], color="Slateblue", alpha=0.6)
                    ax.set_title(f"Area Chart for {selected_col}")
                    st.pyplot(fig)
                else:
                    st.error("This dataset doesn't contain any valid numerical columns for the Area Plot.")

            if st.checkbox("Show Scatter Plot"):
                selected_x = st.selectbox("Select X-axis Column", df.select_dtypes(include=np.number).columns)
                selected_y = st.selectbox("Select Y-axis Column", df.select_dtypes(include=np.number).columns)
                if selected_x and selected_y:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df[selected_x], df[selected_y])
                    ax.set_title(f"Scatter Plot of {selected_x} vs {selected_y}")
                    ax.set_xlabel(selected_x)
                    ax.set_ylabel(selected_y)
                    st.pyplot(fig)
                else:
                    st.error("This dataset doesn't contain any valid numerical columns for the Scatter Plot.")

            if st.checkbox("Show Heatmap"):
                selected_cols = st.multiselect("Select Categorical Columns", df.select_dtypes(include='object').columns)
                if len(selected_cols) >= 2:
                    contingency_table = pd.crosstab(df[selected_cols[0]], df[selected_cols[1]])
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', ax=ax)
                    ax.set_title(f"Heatmap for {selected_cols[0]} vs {selected_cols[1]}")
                    st.pyplot(fig)
                elif len(selected_cols) == 0:
                    st.warning("Please select at least two categorical columns to create a heatmap.")
                else:
                    st.error("This dataset doesn't contain any valid categorical columns to create a heatmap.")
            if st.checkbox("Show Violin Plot"):
                selected_cat_col = st.selectbox("Select Categorical Column", df.select_dtypes(include='object').columns)
                selected_num_col = st.selectbox("Select Numeric Column", df.select_dtypes(include=np.number).columns)
                if selected_cat_col and selected_num_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.violinplot(x=df[selected_cat_col], y=df[selected_num_col], ax=ax)
                    ax.set_title(f"Violin Plot of {selected_num_col} by {selected_cat_col}")
                    st.pyplot(fig)
                elif not selected_cat_col:
                    st.error("Please select a valid categorical column for the x-axis.")
                elif not selected_num_col:
                    st.error("Please select a valid numerical column for the y-axis.")

            if st.checkbox("Show Density Plot"):
                selected_col = st.selectbox("Select Column for Density Plot", df.select_dtypes(include=np.number).columns)
                if selected_col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.kdeplot(df[selected_col], shade=True, ax=ax)
                    ax.set_title(f"Density Plot for {selected_col}")
                    st.pyplot(fig)
                else:
                    st.error("This dataset doesn't contain any valid numerical columns for the Density Plot.")
        elif t == 'Data Transformation':
            df = data_transformation(df)
            st.session_state.df = df

        # Option to download the modified dataset
        if st.sidebar.button("Download Modified Dataset"):
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "modified_dataset.csv", "text/csv", key='download-csv')




    else:
        st.error("Please upload a dataset first.")

if __name__ == '__main__':
    main()
