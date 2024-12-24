import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import streamlit as st # type: ignore
import re
import subprocess
import plotly.express as px # type: ignore

def plot_graph(data, title):
    # Create a Plotly figure
    fig = px.line(data, x=data.index, y=data.columns, title=title)

    # Update the layout for better readability
    fig.update_layout(xaxis_title='Channels', yaxis_title='Values', legend_title='Columns')
    fig.update_traces(line=dict(dash='dash'), selector=dict(name='limit'))

    return fig
# function to run the dataframe selection inn extracted_dataframe 
def process_dataframe(df):
    # Extract information from the DataFrame
    # channel_values = df.iloc[1, 1::2].apply(lambda x: int(str(x)[-2:])0] else int(str(x)[-1:]).tolist()  # Row 1, odd columns
    channel_values = df.iloc[1, 1::2].tolist()  # Extract channel values
    channel_values = [(str(x)[-1]) if str(x)[-2] == '_' else (str(x)[-2:]) for x in channel_values]  # Extract last digit if last two are '_1', else extract last two digits
  
    snb_values = df.iloc[2, 1::2].tolist()  # Row 1, even columns
    snb_values = [(str(x)[-3:-1]) if '*' in str(x) else (str(x)) for x in snb_values]

    limit_values = df.iloc[2, 2::2].tolist()  # Row 2, even columns

    # Create a new DataFrame
    new_dataframe = pd.DataFrame({'channel': channel_values, 'S/N (B)': snb_values, 'limit': limit_values})

    return new_dataframe
def process_html_file32(file_path):
        # If your HTML is in a local file
    file_content = file_path.getvalue()

#     # Process the content (modify this part based on your existing logic)
#     # For example, assuming file_content is a string containing HTML:
#     df = pd.read_html(StringIO(file_content))[0]
#     with open(file_path, 'r', encoding='utf-8') as file:
#         html_content = file.read()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(file_content, 'html.parser')

    # Find all tables in the HTML
    tables = soup.find_all('table')

    # validation 
    
    
    # Calculate the total number of tables
    total_tables = len(tables)

# Define the gap and starting index
    gap = 3
    starting_index = 2

    # Define the indices of the tables you want to retrieve
    table_indices_to_retrieve = list(range(starting_index, total_tables , gap))

    # Define the indices of the tables you want to retrieve (0-based index)
    # table_indices_to_retrieve = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83]

#     for i, table in enumerate(tables):
#         st.write(f"Table {i + 1}:\n")
#         st.write(pd.read_html(str(table))[0])  # Print the entire table
#         st.write("\n" + "=" * 50 + "\n")

    # Extract specific tables into a list of DataFrames
    selected_dataframes = [pd.read_html(str(tables[i]))[0] for i in table_indices_to_retrieve]
    # date need to be added
    
   
    # Now, selected_dataframes list contains DataFrames for the specified tables
   
    # Define a function to extract specific rows from a DataFrame
    def extract_specific_rows(dataframe, row_indices):
        return dataframe.iloc[row_indices, :]
    
    # Define the row indices to extract (0, 1, 24,13)
    row_indices_to_extract = [0, 1, 24,13,12]
    
    # Extract specific rows from each selected DataFrame
    extracted_rows_dataframes = [extract_specific_rows(df, row_indices_to_extract) for df in selected_dataframes]
    
    
    # Now, extracted_rows_dataframes list contains DataFrames with specific rows
    # Create a new DataFrame with 'channel', 'S/N (B)', and 'limit' columns
    new_dataframe = pd.DataFrame(columns=['channel', 'S/N (B)', 'limit'])

    for df in extracted_rows_dataframes:
        processed_df = process_dataframe(df)
        new_dataframe = pd.concat([new_dataframe, processed_df], ignore_index=True)


#  ---------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# all this below is replaced by function ----------------- def def process_dataframe(df):-------------------------------------------------------




#     # Extract information from DataFrame 1 (extracted_rows_dataframes[0])
#     channel_values = extracted_rows_dataframes[0].iloc[1, 1::2].tolist()  # Row 1, odd columns
#     snb_values = extracted_rows_dataframes[0].iloc[2, 1::2].tolist()  # Row 1, even columns
#     limit_values = extracted_rows_dataframes[0].iloc[2, 2::2].tolist()  # Row 2, even columns
    
        
#     # Add information to the new DataFrame
#     new_dataframe['channel'] = channel_values
#     new_dataframe['S/N (B)'] = snb_values
#     new_dataframe['limit'] = limit_values
    

#      # Extract information from DataFrame 3 (extracted_rows_dataframes[1])
#     channel_values = extracted_rows_dataframes[2].iloc[1, 1::2].tolist()  # Row 1, odd columns
#     snb_values = extracted_rows_dataframes[2].iloc[2, 1::2].tolist()  # Row 1, even columns
#     limit_values = extracted_rows_dataframes[2].iloc[2, 2::2].tolist()  # Row 2, even columns
#     new_dataframe = pd.concat([new_dataframe, pd.DataFrame({'channel': channel_values, 'S/N (B)': snb_values, 'limit': limit_values})], ignore_index=True)
#      # Extract information from DataFrame 4 (extracted_rows_dataframes[1])
#     channel_values = extracted_rows_dataframes[3].iloc[1, 1::2].tolist()  # Row 1, odd columns
#     snb_values = extracted_rows_dataframes[3].iloc[2, 1::2].tolist()  # Row 1, even columns
#     limit_values = extracted_rows_dataframes[3].iloc[2, 2::2].tolist()  # Row 2, even columns
#     new_dataframe = pd.concat([new_dataframe, pd.DataFrame({'channel': channel_values, 'S/N (B)': snb_values, 'limit': limit_values})], ignore_index=True)
#     rows_to_move = new_dataframe.iloc[1:4]
#     new_dataframe = pd.concat([new_dataframe, rows_to_move], ignore_index=True).drop(index=range(1,4))

#         # Extract information from DataFrame 2 (extracted_rows_dataframes[1])
#     channel_values = extracted_rows_dataframes[1].iloc[1, 1::2].tolist()  # Row 1, odd columns
#     snb_values = extracted_rows_dataframes[1].iloc[2, 1::2].tolist()  # Row 1, even columns
#     limit_values = extracted_rows_dataframes[1].iloc[2, 2::2].tolist()  # Row 2, even columns
    
#     # Add information to the new DataFrame
#     new_dataframe = pd.concat([new_dataframe, pd.DataFrame({'channel': channel_values, 'S/N (B)': snb_values, 'limit': limit_values})], ignore_index=True)
# #     st.write(new_dataframe)
    # Display the new DataFrame
#     print("New DataFrame:"+"\n")
#     print(new_dataframe)
# ------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------



    # for showing dataframe
    # # Print the extracted rows DataFrames
    # for i, df in enumerate(extracted_rows_dataframes):
    #     print(f"Extracted Rows DataFrame {i + 1}:\n")
    #     print(df)
    #     print("\n" + "=" * 50 + "\n")


    # Plotting the graph
    def extract_numeric_value(s):
        match = re.search(r'\d+', str(s))
        return int(match.group()) if match else None

    # Apply the function to the 'S/N (B)' column
    new_dataframe['limit'] = new_dataframe['limit'].apply(extract_numeric_value)
    # Assuming 'limit' column contains numeric values as strings
    new_dataframe['limit'] = new_dataframe['limit'].astype(str)

    # Remove rows with channel name 'sc'
    new_dataframe = new_dataframe[(new_dataframe['channel'] != 'SC') & (new_dataframe['channel'] != 'C')  & (new_dataframe['channel'] != 'TC')& (new_dataframe['channel'] != 'oc')]



    new_dataframe['channel'] = new_dataframe['channel'].astype(float)
    
    # Set 'channel' column as the index
    new_dataframe.set_index('channel', inplace=True)
    # Sort the index in ascending order
    new_dataframe = new_dataframe.sort_index(ascending=True)
    
    # Display the modified DataFrame
#     print("Modified DataFrame:")
#     print(new_dataframe)
    

    # # Plot the first column on the y-axis
    

    # # Show the plot
    
    # st.set_option('deprecation.showPyplotGlobalUse', True)
    
    #  # Display the plot in Streamlit
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.pyplot(fig)

    # # Display the DataFrame next to the plot
    # with col2:
    #     st.write("### Data Frame")
    #     st.markdown(f'<div style="overflow: auto; height: {fig.get_figheight()}px;">', unsafe_allow_html=True)
    #     st.dataframe(new_dataframe.style.set_table_styles([{'selector': 'table', 'props': [('max-height', '500px')]}]))
       # # Extract the value from extracted_rows_dataframes[0], row 0, column 1
    
    plot_title = extracted_rows_dataframes[0].iloc[0, 1]
    title =extracted_rows_dataframes[0].iloc[3, 1]
    actual_12nc =extracted_rows_dataframes[0].iloc[4, 1]
    file_name=file_path.name
    coil_info={}
    coil_info['plot_title'] = extracted_rows_dataframes[0].iloc[0, 1]
    coil_info['Original_serial_nr'] = extracted_rows_dataframes[0].iloc[3, 1]
    coil_info['actual_12nc'] = extracted_rows_dataframes[0].iloc[4, 1]
    coil_info['file_name'] = file_path.name
    
    match = re.search(r'<td.*?>Date</td><td.*?>(.*?)</td>', str(soup))
    if match:
        coil_info['date_time'] = match.group(1)
    else:
        coil_info['date_time'] = None  # or some default value
    
    return new_dataframe,plot_title,title,coil_info



# function to run the dataframe selection inn extracted_dataframe 


def process_html_file2(file_path):
        # If your HTML is in a local file
    file_content = file_path.getvalue()

#     # Process the content (modify this part based on your existing logic)
#     # For example, assuming file_content is a string containing HTML:
#     df = pd.read_html(StringIO(file_content))[0]
#     with open(file_path, 'r', encoding='utf-8') as file:
#         html_content = file.read()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(file_content, 'html.parser')

    # Find all tables in the HTML
    tables = soup.find_all('table')

    # validation 
    
    # Calculate the total number of tables
    total_tables = len(tables)

# Define the gap and starting index
    gap = 3
    starting_index = 2

    # Define the indices of the tables you want to retrieve
    table_indices_to_retrieve = list(range(starting_index, total_tables , gap))

    
    # Define the indices of the tables you want to retrieve (0-based index)
    # table_indices_to_retrieve = [2, 5, 8,11]
#     for i, table in enumerate(tables):
#         st.write(f"Table {i + 1}:\n")
#         st.write(pd.read_html(str(table))[0])  # Print the entire table
#         st.write("\n" + "=" * 50 + "\n")

    # Extract specific tables into a list of DataFrames
    selected_dataframes = [pd.read_html(str(tables[i]))[0] for i in table_indices_to_retrieve]
    
    # Now, selected_dataframes list contains DataFrames for the specified tables

    # Define a function to extract specific rows from a DataFrame
    def extract_specific_rows(dataframe, row_indices):
        return dataframe.iloc[row_indices, :]
    
    # Define the row indices to extract (0, 1, 24,13)
    row_indices_to_extract = [0, 1, 24,13]
    
    # Extract specific rows from each selected DataFrame
    extracted_rows_dataframes = [extract_specific_rows(df, row_indices_to_extract) for df in selected_dataframes]
    
    
    # Now, extracted_rows_dataframes list contains DataFrames with specific rows
    # Create a new DataFrame with 'channel', 'S/N (B)', and 'limit' columns
    new_dataframe = pd.DataFrame(columns=['channel', 'S/N (B)', 'limit'])


    for df in extracted_rows_dataframes:
        processed_df = process_dataframe(df)
        new_dataframe = pd.concat([new_dataframe, processed_df], ignore_index=True)
    
    # # Extract information from DataFrame 1 (extracted_rows_dataframes[0])
    # channel_values = extracted_rows_dataframes[0].iloc[1, 1::2].tolist()  # Row 1, odd columns
    # snb_values = extracted_rows_dataframes[0].iloc[2, 1::2].tolist()  # Row 1, even columns
    # limit_values = extracted_rows_dataframes[0].iloc[2, 2::2].tolist()  # Row 2, even columns
    
    # # Add information to the new DataFrame
    # new_dataframe['channel'] = channel_values
    # new_dataframe['S/N (B)'] = snb_values
    # new_dataframe['limit'] = limit_values
    

    #  # Extract information from DataFrame 3 (extracted_rows_dataframes[1])
    # channel_values = extracted_rows_dataframes[2].iloc[1, 1::2].tolist()  # Row 1, odd columns
    # snb_values = extracted_rows_dataframes[2].iloc[2, 1::2].tolist()  # Row 1, even columns
    # limit_values = extracted_rows_dataframes[2].iloc[2, 2::2].tolist()  # Row 2, even columns
    # new_dataframe = pd.concat([new_dataframe, pd.DataFrame({'channel': channel_values, 'S/N (B)': snb_values, 'limit': limit_values})], ignore_index=True)
    #  # Extract information from DataFrame 4 (extracted_rows_dataframes[1])
    # channel_values = extracted_rows_dataframes[3].iloc[1, 1::2].tolist()  # Row 1, odd columns
    # snb_values = extracted_rows_dataframes[3].iloc[2, 1::2].tolist()  # Row 1, even columns
    # limit_values = extracted_rows_dataframes[3].iloc[2, 2::2].tolist()  # Row 2, even columns
    # new_dataframe = pd.concat([new_dataframe, pd.DataFrame({'channel': channel_values, 'S/N (B)': snb_values, 'limit': limit_values})], ignore_index=True)
    # rows_to_move = new_dataframe.iloc[1:4]
    # new_dataframe = pd.concat([new_dataframe, rows_to_move], ignore_index=True).drop(index=range(1,4))

    #     # Extract information from DataFrame 2 (extracted_rows_dataframes[1])
    # channel_values = extracted_rows_dataframes[1].iloc[1, 1::2].tolist()  # Row 1, odd columns
    # snb_values = extracted_rows_dataframes[1].iloc[2, 1::2].tolist()  # Row 1, even columns
    # limit_values = extracted_rows_dataframes[1].iloc[2, 2::2].tolist()  # Row 2, even columns
    
    # # Add information to the new DataFrame
    # new_dataframe = pd.concat([new_dataframe, pd.DataFrame({'channel': channel_values, 'S/N (B)': snb_values, 'limit': limit_values})], ignore_index=True)
#     st.write(new_dataframe)
    # Display the new DataFrame
#     print("New DataFrame:"+"\n")
#     print(new_dataframe)

    # for showing dataframe
    # # Print the extracted rows DataFrames
    # for i, df in enumerate(extracted_rows_dataframes):
    #     print(f"Extracted Rows DataFrame {i + 1}:\n")
    #     print(df)
    #     print("\n" + "=" * 50 + "\n")


    # Plotting the graph
    def extract_numeric_value(s):
        match = re.search(r'\d+', str(s))
        return int(match.group()) if match else None

    # Apply the function to the 'S/N (B)' column
    new_dataframe['limit'] = new_dataframe['limit'].apply(extract_numeric_value)
    # Assuming 'limit' column contains numeric values as strings
    new_dataframe['limit'] = new_dataframe['limit'].astype(str)


    new_dataframe = new_dataframe[(new_dataframe['channel'] != 'SC') & (new_dataframe['channel'] != 'HDNK_QBCloc')& (new_dataframe['channel'] != 'HDNK_C')& (new_dataframe['channel'] != 'HEAD_C') & (new_dataframe['channel'] != 'HEAD_QBCloc')  & (new_dataframe['channel'] != 'oc')& (new_dataframe['channel'] != 'C')]

    # Set 'channel' column as the index
   
    new_dataframe['channel'] = new_dataframe['channel'].astype(float)
    new_dataframe.set_index('channel', inplace=True)
    #  ascending order
    # Sort the index in ascending order
    new_dataframe = new_dataframe.sort_index(ascending=True)
    
    # Display the modified DataFrame
#     print("Modified DataFrame:")
#     print(new_dataframe)
    

    # # Plot the first column on the y-axis
    

    # # Show the plot
    
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    
    #  # Display the plot in Streamlit
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.pyplot(fig)

    # # Display the DataFrame next to the plot
    # with col2:
    #     st.write("### Data Frame")
    #     st.markdown(f'<div style="overflow: auto; height: {fig.get_figheight()}px;">', unsafe_allow_html=True)
    #     st.dataframe(new_dataframe.style.set_table_styles([{'selector': 'table', 'props': [('max-height', '500px')]}]))
       # # Extract the value from extracted_rows_dataframes[0], row 0, column 1
    plot_title = extracted_rows_dataframes[0].iloc[0, 1]
    title =extracted_rows_dataframes[0].iloc[3, 1]
    
    coil_info={}
    coil_info['Title'] = extracted_rows_dataframes[0].iloc[0, 1]
    coil_info['Original_serial_nr'] = extracted_rows_dataframes[0].iloc[3, 1]
    coil_info['actual_12nc'] = extracted_rows_dataframes[0].iloc[4, 1]
    coil_info['file_name'] = file_path.name
    
    match = re.search(r'<td.*?>Date</td><td.*?>(.*?)</td>', str(soup))
    if match:
        coil_info['date_time'] = match.group(1)
    else:
        coil_info['date_time'] = None  # or some default value
    return new_dataframe,plot_title,title,coil_info

def process_html_file(file_path):
    # If your HTML is in a local file
    file_content = file_path.getvalue()

#     # Process the content (modify this part based on your existing logic)
#     # For example, assuming file_content is a string containing HTML:
#     df = pd.read_html(StringIO(file_content))[0]
#     with open(file_path, 'r', encoding='utf-8') as file:
#         html_content = file.read()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(file_content, 'html.parser')

    # Find all tables in the HTML
    tables = soup.find_all('table')
    
#     validation 
    if len(tables) < 9:
        st.warning("not 8 channel file")
        return None, None



    # Calculate the total number of tables
    total_tables = len(tables)

# Define the gap and starting index
    gap = 3
    starting_index = 2

    # Define the indices of the tables you want to retrieve
    table_indices_to_retrieve = list(range(starting_index, total_tables - 2, gap))

    # Define the indices of the tables you want to retrieve (0-based index)
    # table_indices_to_retrieve = [2, 5, 8]

    # Extract specific tables into a list of DataFrames
    selected_dataframes = [pd.read_html(str(tables[i]))[0] for i in table_indices_to_retrieve]
   
    # Now, selected_dataframes list contains DataFrames for the specified tables

    # Define a function to extract specific rows from a DataFrame
    def extract_specific_rows(dataframe, row_indices):
        return dataframe.iloc[row_indices, :]

    # Define the row indices to extract (0, 1, 24)
    row_indices_to_extract = [0, 1, 24,13]

    # Extract specific rows from each selected DataFrame
    extracted_rows_dataframes = [extract_specific_rows(df, row_indices_to_extract) for df in selected_dataframes]
    
    # Now, extracted_rows_dataframes list contains DataFrames with specific rows
    # Create a new DataFrame with 'channel', 'S/N (B)', and 'limit' columns
    new_dataframe = pd.DataFrame(columns=['channel', 'S/N (B)', 'limit'])

    # Extract information from DataFrame 1 (extracted_rows_dataframes[0])
    channel_values = extracted_rows_dataframes[0].iloc[1, 1::2].tolist()  # Row 1, odd columns
    snb_values = extracted_rows_dataframes[0].iloc[2, 1::2].tolist()  # Row 1, even columns
    limit_values = extracted_rows_dataframes[0].iloc[2, 2::2].tolist()  # Row 2, even columns

    # Add information to the new DataFrame
    new_dataframe['channel'] = channel_values
    new_dataframe['S/N (B)'] = snb_values
    new_dataframe['limit'] = limit_values

    # Extract information from DataFrame 2 (extracted_rows_dataframes[1])
    channel_values = extracted_rows_dataframes[1].iloc[1, 1::2].tolist()  # Row 1, odd columns
    snb_values = extracted_rows_dataframes[1].iloc[2, 1::2].tolist()  # Row 1, even columns
    limit_values = extracted_rows_dataframes[1].iloc[2, 2::2].tolist()  # Row 2, even columns
    
    # Add information to the new DataFrame
    new_dataframe = pd.concat([new_dataframe, pd.DataFrame({'channel': channel_values, 'S/N (B)': snb_values, 'limit': limit_values})], ignore_index=True)

    # Display the new DataFrame
#     print("New DataFrame:"+"\n")
#     print(new_dataframe)

    # for showing dataframe
    # # Print the extracted rows DataFrames
    # for i, df in enumerate(extracted_rows_dataframes):
    #     print(f"Extracted Rows DataFrame {i + 1}:\n")
    #     print(df)
    #     print("\n" + "=" * 50 + "\n")


    # Plotting the graph
    def extract_numeric_value(s):
        match = re.search(r'\d+', str(s))
        return int(match.group()) if match else None

    # Apply the function to the 'S/N (B)' column
    new_dataframe['limit'] = new_dataframe['limit'].apply(extract_numeric_value)
    # Assuming 'limit' column contains numeric values as strings
    new_dataframe['limit'] = new_dataframe['limit'].astype(str)

    # Set 'channel' column as the index
    new_dataframe['channel'] = new_dataframe['channel'].astype(float)
    
    new_dataframe.set_index('channel', inplace=True)
    # Display the modified DataFrame
#     print("Modified DataFrame:")
#     print(new_dataframe)
    

    # # Plot the first column on the y-axis
    

    # # Show the plot
    
    # st.set_option('deprecation.showPyplotGlobalUse', True)
    
    #  # Display the plot in Streamlit
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.pyplot(fig)

    # # Display the DataFrame next to the plot
    # with col2:
    #     st.write("### Data Frame")
    #     st.markdown(f'<div style="overflow: auto; height: {fig.get_figheight()}px;">', unsafe_allow_html=True)
    #     st.dataframe(new_dataframe.style.set_table_styles([{'selector': 'table', 'props': [('max-height', '500px')]}]))
       # # Extract the value from extracted_rows_dataframes[0], row 0, column 1
    plot_title = extracted_rows_dataframes[0].iloc[0, 1]
    title =extracted_rows_dataframes[0].iloc[3, 1]
    # st.dataframe(new_dataframe.style.set_table_styles([{'selector': 'table', 'props': [('max-height', '500px')]}]))
    
    coil_info={}
    coil_info['title'] = extracted_rows_dataframes[0].iloc[0, 1]
    coil_info['Original_serial_nr'] = extracted_rows_dataframes[0].iloc[3, 1]
    coil_info['actual_12nc'] = extracted_rows_dataframes[0].iloc[4, 1]
    coil_info['file_name'] = file_path.name
    
    match = re.search(r'<td.*?>Date</td><td.*?>(.*?)</td>', str(soup))
    if match:
        coil_info['date_time'] = match.group(1)
    else:
        coil_info['date_time'] = None  # or some default value
    return new_dataframe,plot_title,title,coil_info

    

   



# List of file paths
# file_paths = ['C:\\Users\\320181471\\OneDrive - Philips\\Documents\\Knee 8ch 3.0T\\302775693_459801290632_011.htm','C:\\Users\\320181471\\OneDrive - Philips\\Documents\\Knee 8ch 3.0T\\302775693_459801290632_011.htm']

# # Process each file
# for file_path in file_paths:
#     process_html_file(file_path)
 

# highlighting the crossing values of coil:

def highlight_crossing_limits(df):
    """Highlights cells where values are greater than 'limit' and returns pass/fail lists, skipping 'channel' column."""
    numeric_cols = [col for col in df.columns if col not in ['channel','limit']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
    
    styled_df = df.style.format("{:.0f}", subset=numeric_cols)  # Format all numerical data.
    st.write(styled_df)
    passed_cols = []
    failed_cols = []

    num_rows = len(df)  # Get the number of rows in the DataFrame
    highlight_df = pd.DataFrame('', index=df.index, columns=df.columns)
    
    for col in df.columns:
        if col not in ['limit', 'channel']:  # Skip 'limit' and 'channel' columns
            try:
                # Compare with 'limit' column, handling potential errors
                comparison_result = pd.to_numeric(df[col], errors='coerce') < pd.to_numeric(df['limit'], errors='coerce') 
                
                
                # Reshape comparison_result to match DataFrame shape
                comparison_result = comparison_result.to_numpy().reshape(num_rows, 1)  # Reshape to (num_rows, 1)
               
                # Apply highlighting if comparison is True
                styled_df = styled_df.apply(lambda x: ['background-color: green' if  v else 'background-color: red' for v in comparison_result],axis=0) 
                # highlight_df[col] = comparison_result.apply(lambda x: 'background-color: red' if x else '')                                                                                                            
                # Add to pass/fail lists based on any crossing limit in the column
                if comparison_result.any():
                    failed_cols.append(col)
                else:
                    passed_cols.append(col)
            except (TypeError, ValueError):
                # Handle cases where comparison is not possible
                pass  
    # styled_df = styled_df.apply(lambda x: highlight_df[x.name], axis=1)    
    return styled_df, passed_cols, failed_cols


# Main Streamlit app
def main():
    st.set_page_config(layout="wide")
   
    # Main content in the main area
    st.title('spt report plot')
    # Function selection dropdown in the sidebar
    selected_function = st.sidebar.selectbox('Select Function', ['8 channel', '16 channel','32 channel'], key="function_select")


    # File upload widget with multiple file support
    
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    
    # List to store individual DataFrames for 'S/N (B)'
    uploaded_files = st.sidebar.file_uploader("Choose multiple files", type=["html"], accept_multiple_files=True, key=st.session_state["file_uploader_key"])
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
    if st.sidebar.button("clear all"):
        st.session_state["file_uploader_key"] += 1
        st.experimental_rerun()
   
     
   
    
    # Check if the function selection has changed
    snb_dataframes = pd.DataFrame()
    final_dataframe = pd.DataFrame(columns=['limit'])
    k=0
    all_coil_info = []
    plot_title=" "
    if uploaded_files:
        
        for i,uploaded_file in enumerate(uploaded_files):
            try:
                # Display the modified DataFrame
                
                # Process each uploaded HTML file based on the selected function
                if selected_function == '8 channel':
                    
                    df, plot_title,title,coil_info = process_html_file(uploaded_file)
                    snb_dataframes[f'coil-{title}']=df['S/N (B)']
                    all_coil_info.append(coil_info)
                    if i == 0:
                        st.write(f"Modified DataFrame for {selected_function} {plot_title}")
                #     # Append 'channel' and 'limit' columns to the final DataFrame
                        final_dataframe['limit'] =  df['limit']

                elif selected_function == '32 channel':
                    
                    df, plot_title,title,coil_info = process_html_file32(uploaded_file)
                    snb_dataframes[f'coil-{title}-{i}']=df['S/N (B)']
                    coil_info['Original_serial_nr']+="-"+str(i)
                    all_coil_info.append(coil_info)
                    if i == 0:
                        st.write(f"Modified DataFrame for {selected_function} {plot_title}")
                #     # Append 'channel' and 'limit' columns to the final DataFrame
                        final_dataframe['limit'] =  df['limit']
                        
                        
                    
                else:
                    df, plot_title, title,coil_info = process_html_file2(uploaded_file)
                    snb_dataframes[f'coil-{title}']=df['S/N (B)']
                    all_coil_info.append(coil_info)
                    if i == 0:
                    #     # Append 'channel' and 'limit' columns to the final DataFrame
                        st.write(f"Modified DataFrame for {selected_function} {plot_title}")
                        final_dataframe['limit'] =  df['limit']
                        k=1
                       

               # Process each uploaded HTML file
# #                 df,plot_title = process_html_file(uploaded_file)
#                  # Append 'S/N (B)' column to the list
#                 # snb_dataframes.append(df['S/N (B)'])
#                 snb_dataframes[f'S/N (B)-{i + 1}']=df['S/N (B)']
               
#                 # # Rename the last appended column to 'S/N (B)-i' where i is the file number
#                 # snb_dataframes[-1] = snb_dataframes[-1].rename(f'S/N (B)-{i + 1}')

#                 if i == 0:
#                 #     # Append 'channel' and 'limit' columns to the final DataFrame
#                     final_dataframe['limit'] =  df['limit']

                
               
            except Exception as e:
                st.error(f"An error occurred for {uploaded_file.name}: {e}")
            
        
        # Concatenate the list of DataFrames into a single DataFrame
        final_dataframe = pd.concat([final_dataframe, snb_dataframes], axis=1)
        # for column in final_dataframe:
        #     final_dataframe[column] = pd.to_numeric(final_dataframe[column], errors='coerce')
        st.dataframe(final_dataframe.style.set_table_styles([{'selector': 'table', 'props': [('max-height', '500px')]}]))
        # # Create a figure using Plotly Express
        # fig, px = plt.subplots(figsize=(10, 6))
        # fig = px.line(final_dataframe, x=final_dataframe.index, y=final_dataframe.columns, title='S/N (B) and Limit Visualization')

        # # Update the layout for better readability
        # fig.update_layout(xaxis_title='Channels', yaxis_title='Values', legend_title='Columns')
    

        # Create a Plotly figure
        # fig = plt.figure(figsize=(10, 6))
        # for column in final_dataframe.columns:
        #     if 'S/N (B)' in column or 'limit' in column:
        #         plt.plot(final_dataframe['channel'], final_dataframe[column], label=column)

        # # Update the layout for better readability
        # plt.xlabel('Channels')
        # plt.ylabel('Values')
        # plt.title('S/N (B) and Limit Visualization')
        # plt.legend()
        # Assuming 'channel' is the index in final_dataframe
        fig = px.line(final_dataframe, x=final_dataframe.index, y=final_dataframe.columns, title=f'S/N and Limit Visualization for {plot_title}')

        # Update the layout for better readability
        fig.update_layout(xaxis_title='Channels', yaxis_title='Values', legend_title='Columns')
        # Make the 'limit' column appear as a dashed line
        fig.update_traces(line=dict(dash='dash'), selector=dict(name='limit'))

        
            
            
        # Display the Plotly figure
        st.plotly_chart(fig)
        
        if k==1:
            base_data = final_dataframe[final_dataframe.index <= 8]
            neck_data = final_dataframe[final_dataframe.index > 8]

            
            # User selection for part to include in the plot
            selected_part = st.selectbox('Select Head / Neck Base Part', ['Base', 'Neck'], key="part_select")

            # Plot based on user selection
            if selected_part == 'Base':
                st.plotly_chart(plot_graph(base_data, f'S/N and Limit Visualization for {plot_title} - Base'))
            elif selected_part == 'Neck':
                st.plotly_chart(plot_graph(neck_data, f'S/N and Limit Visualization for {plot_title} - Neck'))

            
         # Create DataFrame from the list of dictionaries
        coil_info_df = pd.DataFrame(all_coil_info) 
        st.write(coil_info_df)  # Display the DataFrame

        # If your DataFrame has a column named 'limit', you may want to exclude it from the calculation
        dynamic_columns = df.columns[df.columns != 'limit']
        for column in dynamic_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        # Calculate the average row-wise for dynamic columns
        df[f'avg_{plot_title}'] = df[dynamic_columns].mean(axis=1)

        # Create a new DataFrame with relevant columns
        avg_dataframe = df[['limit', f'avg_{plot_title}']]
        
        st.write(avg_dataframe)
        styled_df, passed_cols, failed_cols = highlight_crossing_limits(final_dataframe)

        st.dataframe(styled_df)
        st.write(f"Passed Columns: {passed_cols}")
        st.write(f"Failed Columns: {failed_cols}")
        
    
        
        


if __name__ == '__main__':
    main()