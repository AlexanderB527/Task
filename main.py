import numpy as np

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN
from keras.preprocessing import sequence

# Ограничиваем количество слов в датасете IMDB
max_features = 10000
maxlen = 500
batch_size = 32

# Загрузка данных
print('Загрузка данных...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'обучающих последовательностей')
print(len(x_test), 'тестовых последовательностей')

print('Обучающие данные перед обработкой:', x_train[0])

# Ограничиваем размер последовательностей и заполняем недостающие данные
print('Подготовка данных...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('Обучающие данные после обработки:', x_train[0])

# Создаем модель RNN
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# Компилируем модель
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Обучаем модель
print('Обучение модели...')
model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2)

# Оцениваем модель
loss, accuracy = model.evaluate(x_test, y_test)
print('Точность на тестовых данных:', accuracy)