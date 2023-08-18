from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Model eğitimi ve tahmin fonksiyonları
def train_model(train_data, exog_train):
    # Model parametrelerini belirleme
    p = 2  # Örnek: AR(p)
    d = 1  # Örnek: I(d)
    q = 0  # Örnek: MA(q)
    P = 0  # Örnek: Seasonal AR(P)
    D = 1  # Örnek: Seasonal I(D)
    Q = 0  # Örnek: Seasonal MA(Q)
    s = 14  # Örnek: Sezonun uzunluğu (haftada kaç gözlem)
    
    # SARIMA modelini oluşturma
    model = SARIMAX(train_data, exog=exog_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
    
    # Modeli eğitme
    trained_model = model.fit(disp=False)
    
    return trained_model

def make_forecast(model, exog_forecast, forecast_period):
    # Tahmin yapılacak
    forecast = model.get_forecast(steps=len(forecast_period), exog=exog_forecast)
    forecasted_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    
    # Tahmin sonuçlarını bir DataFrame olarak döndür
    forecast_df = pd.DataFrame({
        'Date': forecast_period,
        'Forecast': forecasted_values,
        'Lower_CI': forecast_conf_int.iloc[:, 0],
        'Upper_CI': forecast_conf_int.iloc[:, 1]
    })
    
    return forecast_df

# Örnek veri (örnek veriyi modelinizi eğitmek için gerçek verinizle değiştirin)
data = pd.read_csv("data.csv", parse_dates=["date"], index_col="date").asfreq("D")

# Veriyi temizleme
data['temperature'].fillna(method='ffill', inplace=True)
upper_threshold = 114
data['orders'] = data['orders'].apply(lambda x: x if x < upper_threshold else np.nan)
data['orders'].fillna(method='ffill', inplace=True)
data['temperature_abs'] = data['temperature'].abs()

# Ölçeklendirilmiş değer sütunu oluşturma
scaler = StandardScaler()
data['temperature_scaled'] = scaler.fit_transform(data[['temperature']])

train_data = data['orders']
exog_train = data[['temperature_scaled', 'media_spend']]

# Modeli eğit
model = train_model(train_data, exog_train)

@app.route('/forecast', methods=['GET'])
def forecast():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    forecast_period = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Exog değerleri belirle (örnek olarak)
    exog_forecast = data.loc[forecast_period][['temperature_scaled', 'media_spend']]
    
    # Tahmin yap
    forecasted_values = make_forecast(model, exog_forecast, forecast_period)
    
    result = {
        'forecast_period': forecast_period.strftime('%Y-%m-%d'),
        'forecast_values': forecasted_values.to_dict(orient='records')
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)