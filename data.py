import xml.etree.ElementTree as ET
import pandas as pd
import re

from typing import *


def extract_data(path: str) -> pd.DataFrame:
    """
    функция для извлечения данных из xml
    """
    tree = ET.parse(path)
    root = tree.getroot()
    DataFrame = dict()
    database = root.findall('database')[0]
    DataFrame_columns = list()

    for idx, table in enumerate(database.findall('table')):
        for column in table.findall('column'):
            DataFrame[column.attrib['name']] = list()
            DataFrame_columns.append(column.attrib['name'])
        if idx == 0:
            break

    for table in database.findall('table'):
        for column in table.findall('column'):
            DataFrame[column.attrib['name']].append(column.text)

    data = pd.DataFrame(DataFrame, columns=DataFrame_columns)
    return data


def extract_text_features(
        data: pd.DataFrame, columns_with_labels: List[str]
    ) -> pd.DataFrame:
    """
    функция для первичной обработки текста от лишних символов
    """
    extracted_data = dict()
    extracted_data['text'] = list()
    extracted_data['0class'] = list()
    extracted_data['1class'] = list()

    for idx in range(len(data)):
        row = data.iloc[idx, :]
        banks_review = row[columns_with_labels]
        unique_labels = set(banks_review)
        unique_labels.remove('NULL')

        # убираем все ненужные знаки
        filtered_text = re.sub('http[A-z|:|.|/|0-9]*', '', row['text']).strip()
        filtered_text = re.sub('@\S*', '', filtered_text).strip()
        filtered_text = re.sub('#', '', filtered_text).strip()
        new_text = filtered_text

        # сохраняем только уникальные токены (без придатка xml NULL)
        unique_labels = list(unique_labels)
        while len(unique_labels) < 2:
            unique_labels.append(unique_labels[-1])
        extracted_data['text'].append(new_text)
        for idx, label in enumerate(unique_labels):
            text_label = int(label) + 1
            extracted_data[f'{idx}' + 'class'].append(text_label)

    extracted_data = pd.DataFrame(extracted_data)
    
    # возвращаем dataframe
    return extracted_data