import streamlit as st
import pandas as pd
import joblib



# Загружаем модели и кодировщик
lr_model = joblib.load('Desktop/jupyter/lr_model.pkl')
dt_model = joblib.load('Desktop/jupyter/dt_model.pkl')
rf_model = joblib.load('Desktop/jupyter/rf_model.pkl')
xgb_model = joblib.load('Desktop/jupyter/xgb_model.pkl')
encoder = joblib.load('Desktop/jupyter/encoder.pkl')


p_im = """
<style>
	[data-testid="stAppViewContainer"]{
		background-color: #f5d7d73e;
        background-size: cover;
        text-align: center;
        }
</style>"""


st.markdown(p_im,unsafe_allow_html=True)
# Заголовок
st.title('Предсказание цены авиабилета')

# Описание приложения
st.markdown("""
Добро пожаловать в приложение для предсказания цены авиабилетов! 
Введите информацию о вашем рейсе, чтобы узнать предсказанную стоимость билета.
""")

# Разделитель
st.markdown('---')

st.image("https://static.aviasales.com/selene-static/entrypoint/4153d7fd94484074d2bd.png")

# Ввод данных пользователем
st.sidebar.header('Введите данные рейса ✈')

data = pd.read_csv('Desktop/jupyter/flight_data.csv') # Для примера выбора уникальных значений
col1, col2 = st.columns(2)

with col1:
    airline = st.sidebar.selectbox('Авиакомпания', sorted(data['airline'].unique()))
    source_city = st.sidebar.selectbox('Город вылета', sorted(data['source_city'].unique()))
    destination_city = st.sidebar.selectbox('Город прилета', sorted(data['destination_city'].unique()))
    travel_class = st.sidebar.selectbox('Класс', sorted(data['class'].unique()))
    
    
    

with col2:
    departure_time = st.sidebar.selectbox('Время вылета', sorted(data['departure_time'].unique()))
    duration = st.sidebar.number_input('Продолжительность полета (часы)', min_value=0.0, step=0.1)
    days_left = st.sidebar.number_input('Дней до вылета', min_value=0, step=1)

# Преобразование введенных данных
input_data = pd.DataFrame({
    'airline': [airline],
    'source_city': [source_city],
    'destination_city': [destination_city],
    'departure_time': [departure_time],
    'class': [travel_class],
    'duration': [duration],
    'days_left': [days_left]
})

# Преобразуем только категориальные признаки
input_encoded_cat = encoder.transform(input_data[['airline', 'source_city', 'destination_city', 'departure_time', 'class']])

# Создаем финальный датафрейм для предсказания
input_encoded = pd.concat([pd.DataFrame(input_encoded_cat.toarray()), input_data[['duration', 'days_left']].reset_index(drop=True)], axis=1)

# Обновляем названия столбцов для input_encoded
input_encoded.columns = input_encoded.columns.astype(str)

# Кнопка для предсказания
if st.sidebar.button(':red[Предсказать цену]'):
    # Предсказание цены с помощью моделей
    lr_predicted_price = lr_model.predict(input_encoded)[0]
    dt_predicted_price = dt_model.predict(input_encoded)[0]
    rf_predicted_price = rf_model.predict(input_encoded)[0]
    xgb_predicted_price = xgb_model.predict(input_encoded)[0]
    
    # Отображение предсказанных цен
    st.markdown('---')
    st.header(':blue[Предсказанная цена авиабилета:]')
    st.write(f'**(Линейная регрессия):** {lr_predicted_price:.2f} руб.')
    st.write(f'**(Дерево решений):** {dt_predicted_price:.2f} руб.')
    st.write(f'**(Случайный лес):** {rf_predicted_price:.2f} руб.')
    st.write(f'**(XGBoost):** {xgb_predicted_price:.2f} руб.')

    # Графическое представление результатов
    st.markdown('---')
    st.header(':blue[Визуализация]')
    results = pd.DataFrame({
        'Модель': ['Линейная регрессия', 'Дерево решений', 'Случайный лес', 'XGBoost'],
        'Цена': [lr_predicted_price, dt_predicted_price,rf_predicted_price, xgb_predicted_price]
    })
    

    st.bar_chart(results.set_index('Модель'))

# Разделитель
#st.markdown('---')

# Завершающий текст
st.sidebar.markdown("""
Мы надеемся, что наш сервис помог вам. Удачного полета! ✈️
""")

