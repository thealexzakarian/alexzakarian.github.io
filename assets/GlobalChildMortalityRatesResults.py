#Pandas for data preprocessing and merging 
#NumPy for implementing mathematical operations in SA and Linear Regression 
import pandas as pd
import numpy as np


# Load data files 
mortality_africa = pd.read_csv('/Users/alexzakarian/Desktop/Msc Computer Science with AI - York/Applied AI/AAI_2024_Datasets /Child mortality rates_Africa.csv')
mortality_americas = pd.read_csv('/Users/alexzakarian/Desktop/Msc Computer Science with AI - York/Applied AI/AAI_2024_Datasets /Child mortality rates_Americas.csv')
mortality_eastern_med = pd.read_csv('/Users/alexzakarian/Desktop/Msc Computer Science with AI - York/Applied AI/AAI_2024_Datasets /Child mortality rates_Eastern_Mediterranean.csv')
mortality_europe = pd.read_csv('/Users/alexzakarian/Desktop/Msc Computer Science with AI - York/Applied AI/AAI_2024_Datasets /Child mortality rates_Europe.csv')
mortality_south_asia = pd.read_csv('/Users/alexzakarian/Desktop/Msc Computer Science with AI - York/Applied AI/AAI_2024_Datasets /Child mortality rates_South_East_Asia.csv')
mortality_west_pacific = pd.read_csv('/Users/alexzakarian/Desktop/Msc Computer Science with AI - York/Applied AI/AAI_2024_Datasets /Child mortality rates_Western_Pacific.csv')
nutrition_data = pd.read_csv('/Users/alexzakarian/Desktop/Msc Computer Science with AI - York/Applied AI/AAI_2024_Datasets /Infant nutrition data by country.csv')


# Clean the datasets dropping rows with missing values
def clean_data(df):
    df = df.dropna()  
    return df


# clean datasets
mortality_africa = clean_data(mortality_africa)
mortality_americas = clean_data(mortality_americas)
mortality_eastern_med = clean_data(mortality_eastern_med)
mortality_europe = clean_data(mortality_europe)
mortality_south_asia = clean_data(mortality_south_asia)
mortality_west_pacific = clean_data(mortality_west_pacific)
nutrition_data = clean_data(nutrition_data)


# Merge mortality datasets into one dataframe
mortality_data = pd.concat([
    mortality_africa, mortality_americas, mortality_eastern_med,mortality_europe, mortality_south_asia, mortality_west_pacific
])

#forcing year data to be an int
mortality_data['Year']=mortality_data['Year'].astype(int)
nutrition_data['Year']=nutrition_data['Year'].astype(int)

#print(mortality_data.head().to_string())
#print(nutrition_data.head().to_string())
#this was done to confirm the datasets were corrected 

#merging nutrition and mortality data based on Country and Year variables
merged_data = mortality_data.merge(nutrition_data, how='inner', on=['Country', 'Year'])

#print(merged_data.head().to_string())
#this was done to confirm the corrected datasets were merged in one file




#Simluated Annealing function for selection optimizing Linear Regression 

# Function to extract the central numeric value from the string
def extract_central_value(val):
    try:
        # split and take the first part before the bracket and convert to float
        return float(val.split(' ')[0])
    except ValueError:
        return float('None')  # Return None if conversion to float fails

# Functiion to change colunm value to float from %
merged_data['Infants exclusively breastfed for the first six months of life (%)'] = merged_data['Infants exclusively breastfed for the first six months of life (%)'].apply(extract_central_value)

# Function to extract central value of merged 'Under-five mortality rate (per 1000 live births) (SDG 3.2.1) Both sexes' data
merged_data['Under-five mortality rate (per 1000 live births) (SDG 3.2.1) Both sexes'] = merged_data['Under-five mortality rate (per 1000 live births) (SDG 3.2.1) Both sexes'].apply(extract_central_value)

# Depednent and Indendent variables for linear regression model
X = merged_data[['Infants exclusively breastfed for the first six months of life (%)']].values.astype(float)  # making sure X is a float 
y = merged_data[['Under-five mortality rate (per 1000 live births) (SDG 3.2.1) Both sexes']].values.astype(float)  # making sure Y is a float 

# Define the linear regression model
def predict(X, weights):
    return X.dot(weights)

# Mean Squared Error MSE definition 
def mse(weights, X, y):
    predictions = predict(X, weights)
    return np.mean((predictions - y) ** 2)

# Simulated Annealing function for linear regression
def simulated_annealing_linear_regression(X, y, max_iter, initial_temp, cooling_rate):
    
    # Initial solution (random weights for linear regression)
    num_features = X.shape[1] + 1  # We add one for the intercept term
    initial_solution = np.random.randn(num_features, 1)  # Random initial weights
    X_2 = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term to X
    
    # Initial function (MSE)
    current_objective = mse(initial_solution, X_2, y)
    
    # object to store the current best solution
    best_solution = initial_solution
    best_objective = current_objective
    
    # intialization fo initial temperature for SA
    temperature = initial_temp
    
    for i in range(max_iter):
# Create a new potential solution by randomly adjusting the current one
        new_solution = initial_solution + np.random.randn(num_features, 1) * 0.1  
        
        # Evaluate the objective function for the new solution
        new_objective = mse(new_solution, X_2, y)
        
    # Calculate the acceptance probability
        if new_objective < current_objective:
    # Accept the new solution if it's better
            initial_solution = new_solution
            current_objective = new_objective
        else:
            # Accept worse solution 
            acceptance_probability = np.exp((current_objective - new_objective) / temperature)
            if np.random.rand() < acceptance_probability:
                initial_solution = new_solution
                current_objective = new_objective
        
        # Update best solution
        if current_objective < best_objective:
            best_solution = initial_solution
            best_objective = current_objective
        
        # Cooling temperature
        temperature *= cooling_rate
    
    return best_solution, best_objective

# simulated annealing parameters: number of iterations, intial temp, and cooling rate
iteration_limit = 10000  
initial_temp = 10.0  
cooling_rate = 0.99  



# Simulated Annealing with merged dataset for linear regression 
best_weights, best_mse = simulated_annealing_linear_regression(X, y, iteration_limit, initial_temp, cooling_rate)

# Function to print results for linear regression including: intecept, coeffiecnet and best MSE obtained through iterations
print(f"Best Weights (Intercept and Coefficients):\n {best_weights}")
print(f"Best MSE: {best_mse}")