from sklearn.ensemble import RandomForestRegressor

# Define the Random Forest Regressor model
def train_rf_model(X_train_scaled, X_test_scaled, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the regression model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")