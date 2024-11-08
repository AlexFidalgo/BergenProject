import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def predict_rcm_error_with_linear_regression_for_fixed_metric(cons, target_rcm, fixed_gcm, error_metric):
    # Filter data for the specified GCM and metric
    data = cons[(cons['id_GCM'] == fixed_gcm) & (cons['Metric'] == error_metric)]
    
    # Pivot data to have RCMs as columns and gridpoints as rows
    pivot_data = data.pivot_table(index='gp_region', columns='id_RCM', values='mat_vector')

    # Check if we have sufficient data
    if target_rcm not in pivot_data.columns:
        print(f"No data available for target RCM {target_rcm}.")
        return
    
    # Define the target (error of the specific RCM) and features (errors of other RCMs)
    X = pivot_data.drop(columns=target_rcm)  # Features: errors from other RCMs
    y = pivot_data[target_rcm]               # Target: error of the specified RCM

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")

    # Output the coefficients for insight into which RCMs influence the target RCM
    coef_df = pd.DataFrame({'RCM': X.columns, 'Coefficient': model.coef_})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

    print("Top influencing RCMs on the target RCM:")
    print(coef_df.head())

    return model, coef_df
