import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



#load iphone excel file
phone_data = pd.read_excel(r"E:\Data_science_projects\Basic_projects\Iphone_price_prediction\Book1.xlsx")

print(phone_data.head())


#ploting or Visulization
plt.scatter(phone_data['Version'], phone_data['Price'], color = 'blue', label='Actual Data')
#plt.show()


#Prediction future price

m = LinearRegression()
m.fit(phone_data[['Version']], phone_data[['Price']])

#predicting 15 version price
future_version = [[15]]
predicted_price = m.predict(future_version)[0][0]
print(f"Predicted price for iphone version 15: {predicted_price}")

# create prediction line for visualization
x_range = range(phone_data['Version'].min(), 18)
y_pred_line = m.predict(pd.DataFrame(x_range))

#plot regression line
plt.plot(x_range, y_pred_line, color = 'red', label ='Prediction Line')


# Mark the predicted point
plt.scatter(15, predicted_price, color='green', s=100, marker='x', label=f'Predicted (15, {predicted_price:.2f})')

plt.xlabel("iPhone Version")
plt.ylabel("Price")
plt.title("iPhone Price Prediction")
plt.legend()
plt.grid(True)
plt.show()

# Save predictions into Excel
predictions_df = pd.DataFrame({
    "Version": list(x_range),
    "Predicted_Price": y_pred_line.flatten()
})




