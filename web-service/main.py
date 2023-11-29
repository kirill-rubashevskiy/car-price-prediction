# импорты библиотек

# работа с моделью
import dill as pickle

import numpy as np
import pandas as pd
import io
from typing import Union

# работа FastAPI сервиса
import uvicorn
from fastapi import FastAPI, UploadFile, Response, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

import schemas

# загрузка модели
with open('./models/final_model_pipeline.pkl', 'rb') as file:
    final_model_pipeline = pickle.load(file)

# создание сервиса
app = FastAPI()

@app.get("/")
def root():
    '''
    Функция возвращает приветственное сообщение главной страницы сервиса.

    :return: приветственное сообщение
    '''

    return {"message": f"Welcome to the Car Price Prediction Web Service"}

@app.post("/predict_item")
def predict_item(item: schemas.Item) -> float:
    '''
    Функция предсказывает стоимость автомобиля по параметрам, переданным в json формате.

    :param item: параметры автомобиля
    :return: предсказание стоимости автомобиля
    '''

    item_df = pd.DataFrame(jsonable_encoder(item), index=[0])
    prediction = final_model_pipeline.predict(item_df)[0]

    return prediction

@app.post('/predict_items')
def predict_items(file: UploadFile) -> str:
    '''
    Функция предсказывает стоимость автомобилей, параметры которых переданы в формате csv-файла.

    :param file: csv-a f
    :return: предсказание стоимости автомобиля
    '''

    # загрузка и чтение файла с параметрами автомобилей
    content = file.file.read()
    buffer = io.BytesIO(content)
    df = pd.read_csv(buffer, index_col=0)
    file.close()

    # предсказание цен с сохранением результатов в исходный файл
    df['selling_price'] = final_model_pipeline.predict(df)

    # возврат модифицированного исходного файла
    stream = io.StringIO()
    df.to_csv(stream, index = False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                )
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response


# запуск сервиса
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
