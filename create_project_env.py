"""
Скрипт создает необходимые для проекта папки
"""
import os

home = str(os.getcwdb())
folders = ['data', 'models', 'results']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
print(f'Папки под проект созданы в: {home}')
