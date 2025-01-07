import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# GUI Setup
root = tk.Tk()
root.title("Activity Log Data Processor")
root.geometry("900x800")

# Global Variables for GUI and Functions
activity_log = None
user_log = None
component_codes = None
processed_json = None
interaction_counts_month = None
interaction_counts_week = None
statistics = None
is_saved_to_json = False 

# Step 1: Function To Load Initial CSV Files or JSON files Converted from CSV
def load_file(file_type):
    global activity_log, user_log, component_codes
    file_path = filedialog.askopenfilename(filetypes=[("CSV or JSON files", "*.csv *.json")])
    if file_path:
        try:
            # Try Except Block Used to determine the file type is either CSV or JSON based on file extension i.e. .csv or .json
            file_extension = file_path.split('.')[-1].lower()

            if file_type == "activity_log":
                if file_extension == "csv":
                    activity_log = pd.read_csv(file_path, usecols=["User Full Name *Anonymized", "Component", "Action"])
                elif file_extension == "json":
                    activity_log = pd.read_json(file_path)
                messagebox.showinfo("File Loaded", "Activity Log loaded successfully!")

            elif file_type == "user_log":
                if file_extension == "csv":
                    user_log = pd.read_csv(file_path, usecols=["User Full Name *Anonymized", "Date"])
                elif file_extension == "json":
                    user_log = pd.read_json(file_path)
                messagebox.showinfo("File Loaded", "User Log loaded successfully!")

            elif file_type == "component_codes":
                if file_extension == "csv":
                    component_codes = pd.read_csv(file_path)
                elif file_extension == "json":
                    component_codes = pd.read_json(file_path)
                messagebox.showinfo("File Loaded", "Component Codes loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

# Step 2: Function To Convert The Inital CSV Files That Were Loaded to JSON Files 
def save_files_to_json_format():
    global activity_log, user_log, component_codes, is_saved_to_json

    if activity_log is None or user_log is None or component_codes is None:
        messagebox.showwarning("Missing Files", "Please load all required files first!")
        return

    try:
        # Inidcators If File Loading Was Successful Or Not
        activity_log_saved = False
        user_log_saved = False
        component_codes_saved = False

        # Save each loaded CSV file as a JSON File
        activity_log_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Activity Log as JSON")
        if activity_log_path:
            activity_log.to_json(activity_log_path, orient="records", lines=False, indent=4)
            activity_log_saved = True

        user_log_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save User Log as JSON")
        if user_log_path:
            user_log.to_json(user_log_path, orient="records", lines=False, indent=4)
            user_log_saved = True

        component_codes_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Component Codes as JSON")
        if component_codes_path:
            component_codes.to_json(component_codes_path, orient="records", lines=False, indent=4)
            component_codes_saved = True

        # Feedback To Determeine that the Files Were Saved Or Not
        feedback_message = "Files saved as JSON:\n"
        if activity_log_saved:
            feedback_message += f" - Activity Log: {activity_log_path}\n"
        if user_log_saved:
            feedback_message += f" - User Log: {user_log_path}\n"
        if component_codes_saved:
            feedback_message += f" - Component Codes: {component_codes_path}\n"

        if activity_log_saved or user_log_saved or component_codes_saved:
            is_saved_to_json = True
            messagebox.showinfo("Files Saved", feedback_message)
        else:
            messagebox.showwarning("No Files Saved", "No files were saved as JSON!")

# If Files Were Not Saved an Error Message Should Be Sent 
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save files to JSON: {str(e)}")


# Step 3-9: Data Processing, Transformation and Merging Proceseed Data Into One Optimized File 
def process_all_given_data():
    global activity_log, user_log, component_codes, interaction_counts_month, interaction_counts_week

    if activity_log is None or user_log is None or component_codes is None:
        messagebox.showwarning("Missing Files", "Please load all required files first!")
        return

    try:
        # Step 3: Rename Colunms For Consistency For Following Processing Steps
        activity_log.rename(columns={"User Full Name *Anonymized": "User_ID"}, inplace=True)
        user_log.rename(columns={"User Full Name *Anonymized": "User_ID"}, inplace=True)
        messagebox.showinfo("Step 2 Complete", "Columns renamed successfully!")

        # Step 4: Parse the Date column
        user_log['Timestamp'] = pd.to_datetime(user_log['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
        messagebox.showinfo("Step 3 Complete", "Date column parsed successfully!")
    
        # Step 5: Remove "Systems and Folder" Components Which Are Not Needed For Analysis 
        if 'Component' in activity_log.columns:
            activity_log = activity_log[~activity_log['Component'].isin(['System', 'Folder'])]
            messagebox.showinfo("Step 4 Complete", "Unnecessary components removed!")
        else:
            messagebox.showwarning("Missing Data", "The 'Component' column is not in the Activity Log!")

        # Step 6: Merge component codes into activity log
        activity_log = pd.merge(activity_log, component_codes, on="Component", how="left")
        messagebox.showinfo("Step 5 Complete", "Component codes merged successfully!")

        # Step 7: Merge User Log (with Timestamp) To Activity Log
        activity_log = pd.merge(activity_log, user_log[['User_ID', 'Timestamp']], on="User_ID", how="left")
        messagebox.showinfo("Step 6 Complete", "User log merged successfully!")

        # Step 8: Extract The Month and The Week From Activity Log
        activity_log['Month'] = activity_log['Timestamp'].dt.to_period('M')
        activity_log['Week'] = activity_log['Timestamp'].dt.isocalendar().week
        messagebox.showinfo("Step 7 Complete", "Month and Week columns extracted successfully!")

        # Step 9: Get Interaction Counts For Months and Weeks
        global interaction_counts_month, interaction_counts_week
        interaction_counts_month = activity_log.pivot_table(index=["User_ID", "Month"], columns="Component", values="Action", aggfunc="count", fill_value=0).reset_index()

        interaction_counts_week = activity_log.pivot_table(index=["User_ID", "Week"], columns="Component", values="Action", aggfunc="count", fill_value=0).reset_index()

        messagebox.showinfo("Processing Complete", "Data processed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Data processing failed: {str(e)}")

# Step 10: Calculate Statistics For Each Determined Time Point (Semester, Month and Weeks)
def calculate_statistics(data_month, data_week, components):
    stats = {"Monthly": {}, "Weekly": {}, "Semester": {}}
    grouped_by_month = data_month.groupby("Month")
    grouped_by_week = data_week.groupby("Week")

    # Monthly statistics (Mean, Median, Mode)
    for month, group in grouped_by_month:
        monthly_stats = {}
        for comp in components:
            if comp in group:
                values = group[comp]
                monthly_stats[comp] = {"mean": values.mean(), "median": values.median(),"mode": values.mode().iloc[0] if not values.mode().empty else None}
            else:
                monthly_stats[comp] = {"mean": None, "median": None, "mode": None}
        stats["Monthly"][str(month)] = monthly_stats

    # Weekly Statistics (Mean, Median, Mode)
    for week, group in grouped_by_week:
        weekly_stats = {}
        for comp in components:
            if comp in group:
                values = group[comp]
                weekly_stats[comp] = {"mean": values.mean(),"median": values.median(), "mode": values.mode().iloc[0] if not values.mode().empty else None}
            else:
                weekly_stats[comp] = {"mean": None, "median": None, "mode": None}
        stats["Weekly"][week] = weekly_stats

    # Semester Statistics (Mean, Median, Mode)
    semester_data = data_week[components]
    for comp in components:
        if comp in semester_data:
            values = semester_data[comp]
            stats["Semester"][comp] = {"mean": values.mean(),"median": values.median(),"mode": values.mode().iloc[0] if not values.mode().empty else None,}
        else:
            stats["Semester"][comp] = {"mean": None, "median": None, "mode": None}

    return stats

# Display Statistics For Each Interaction Component
def display_all_statistics():
    global interaction_counts_month, interaction_counts_week

    if interaction_counts_month is None or interaction_counts_week is None:
        messagebox.showwarning("No Data", "Please process the data first!")
        return

    components_of_interest = ["Quiz", "Lecture", "Assignment", "Attendence", "Survey"]
    global statistics
    statistics = calculate_statistics(interaction_counts_month, interaction_counts_week, components_of_interest)

    # Display Results in GUI Output Box
    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, "Monthly Statistics:\n")
    for month, stats in statistics["Monthly"].items():
        output_box.insert(tk.END, f"Month: {month}\n")
        for comp, values in stats.items():
            output_box.insert(tk.END, f"  {comp}: Mean={values['mean']}, Median={values['median']}, Mode={values['mode']}\n")

    output_box.insert(tk.END, "\nWeekly Statistics:\n")
    for week, stats in statistics["Weekly"].items():
        output_box.insert(tk.END, f"Week: {week}\n")
        for comp, values in stats.items():
            output_box.insert(tk.END, f"  {comp}: Mean={values['mean']}, Median={values['median']}, Mode={values['mode']}\n")

    output_box.insert(tk.END, "\nSemester Statistics (13 Weeks):\n")
    for comp, values in statistics["Semester"].items():
        output_box.insert(tk.END, f"  {comp}: Mean={values['mean']}, Median={values['median']}, Mode={values['mode']}\n")


# Function To Backup Data To JSON File
# For Some Reason It loads or backs up some files and then it crashes 
# Previously put print statements to locate the error but it solution was not found 
def backup_all_data():
    global activity_log
    if activity_log is None or activity_log.empty:
        messagebox.showwarning("No Data", "Please process the data first!")
        return
    
    save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if save_path:
        try:

            # Converting DataFrame To Dictionary 
            activity_log_dict = activity_log.to_dict(orient="records")
            
            # Write to JSON using the `json` module (avoiding pandas' to_json)
            import json
            with open(save_path, "w") as f:
                json.dump(activity_log_dict, f, indent=4)  # Pretty-print JSON with indentation

            messagebox.showinfo("Backup Complete", "Processed data backed up successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to back up data: {str(e)}")



# Function To Restore Data From Saved Point 
def restore_data():
    global activity_log, interaction_counts_month, interaction_counts_week
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        try:
            activity_log = pd.read_json(file_path, lines=True)
            interaction_counts_month = activity_log.pivot_table(index=["User_ID", "Month"], columns="Component", values="Action", aggfunc="count", fill_value=0).reset_index()

            interaction_counts_week = activity_log.pivot_table(
                index=["User_ID", "Week"], columns="Component", values="Action", aggfunc="count", fill_value=0).reset_index()

            messagebox.showinfo("Restore Complete", "Data restored successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to restore data: {str(e)}")



# Pearson Correlation Analysis Represented by Heatmap
def plot_pearson_correlation_heatmap():
    global interaction_counts_month

    if interaction_counts_month is None:
        messagebox.showwarning("No Data", "Please process the data first!")
        return

    try:
        # Specify Interaction Componenets For Coorelation Analysis
        components = ["Assignment", "Quiz", "Lecture", "Book", "Project", "Course"]

        # Ensure User_ID is Converted To Numeric Format So It Can be Used As A Variable For Coorelation Analysis Against Other Interaction Componenets 
        interaction_counts_month["User_ID_Numeric"] = interaction_counts_month["User_ID"].astype("category").cat.codes

        # Filter and ensure all components are numeric # Remove rows with NaN values
        valid_components = ["User_ID_Numeric"] + [comp for comp in components if comp in interaction_counts_month.columns]
        numeric_data = interaction_counts_month[valid_components].apply(pd.to_numeric, errors='coerce')
        numeric_data = numeric_data.dropna()  

        # Calculate correlation matrix (Pearson Coorelation)
        correlation_matrix = numeric_data.corr()

        # Replace any remaining NaN values with zero
        correlation_matrix.fillna(0, inplace=True)

        # Heatmap Features Which Include: Correlation values, formatting numbers to two decimal places, Color map of heatmap, 
        # Line width Between Cells, Line color, Square-shaped Cells, Annotation Text Size, and Colorbar Adjustment

        plt.figure(figsize=(16, 14))
        heatmap = sns.heatmap(
            correlation_matrix, annot=True,fmt=".2f", cmap="coolwarm", linewidths=0.5,           
            linecolor="white", square=True, annot_kws={"size": 10},cbar_kws={"shrink": 0.8}   
        )

        # Add text annotations for each cell explicitly 
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                value = correlation_matrix.iloc[i, j]
                heatmap.text(
                    j + 0.5, i + 0.5, f"{value:.2f}",
                    ha="center", va="center", color="black")

        # Labels and Titles for Heatmap
        plt.title("Correlation Matrix of User_ID and Component Interactions", fontsize=18, pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.xlabel("Variables", fontsize=14)
        plt.ylabel("Variables", fontsize=14)

        # Heatmap Layout and Adjustment
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot correlation matrix: {str(e)}")



# Stacked Bar Graph Function
def plot_stacked_bargraph():
    global interaction_counts_month

    if interaction_counts_month is None:
        messagebox.showwarning("No Data", "Please process the data first!")
        return
    try:
        # Month has to be in String format to be plotted 
        interaction_counts_month["Month"] = interaction_counts_month["Month"].astype(str)

        # Filter Desired Interaction Componenents 
        components = ["Assignment", "Quiz", "Lecture", "Book", "Project", "Course"]
        valid_components = [comp for comp in components if comp in interaction_counts_month.columns]

        if not valid_components:
            messagebox.showerror("Error", "No valid components for plotting!")
            return

        # Data is then aggregated by Month 
        aggregated_data = interaction_counts_month.groupby("Month")[valid_components].sum().reset_index()

        # Prepare data for plotting
        months = aggregated_data["Month"]
        bar_width = 0.85  # Full-width bars since they're stacked
        bottom = [0] * len(months)

        # Bar Graph Features
        plt.figure(figsize=(14, 8))
        for component in valid_components:
            plt.bar(months, aggregated_data[component], width=bar_width, label=component,bottom=bottom)
            bottom = [i + j for i, j in zip(bottom, aggregated_data[component])]

        # Add labels, title, and legend
        plt.xlabel("Months", fontsize=12)
        plt.ylabel("Total Interactions", fontsize=12)
        plt.title("Monthly User Interactions Across Components", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.legend(title="Components", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Display Bar Graph
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot stacked bar graph: {str(e)}")



# GUI Layout: Buttons For Each Function 
tk.Label(root, text="Activity Log Data Processor", font=("Helvetica", 16)).pack(pady=10)
tk.Button(root, text="Load Activity Log CSV or JSON", command=lambda: load_file("activity_log")).pack(pady=5)
tk.Button(root, text="Load User Log CSV or JSON", command=lambda: load_file("user_log")).pack(pady=5)
tk.Button(root, text="Load Component Codes CSV or JSON", command=lambda: load_file("component_codes")).pack(pady=5)
tk.Button(root, text="Save Loaded CSV Files to JSON", command=save_files_to_json_format).pack(pady=10)
tk.Button(root, text="Process Data", command=process_all_given_data).pack(pady=10)
tk.Button(root, text="Show Statistics", command=display_all_statistics).pack(pady=10)
tk.Button(root, text="Backup Data to JSON", command=backup_all_data).pack(pady=10)
tk.Button(root, text="Restore Data from JSON", command=restore_data).pack(pady=10)
tk.Button(root, text="Correlation Heatmap", command=plot_pearson_correlation_heatmap).pack(pady=10)
tk.Button(root, text="Stacked Bar Graph", command=plot_stacked_bargraph).pack(pady=10)

# Output Box for Statistical Calculation Results (Month, Week, Semester)
output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)
output_box.pack(pady=10)

# Function To Run The Entire GUI
root.mainloop()

