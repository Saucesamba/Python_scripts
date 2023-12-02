import re

def extract_abbreviations(file_path):
    abbreviations = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # Используем регулярное выражение для поиска аббревиатур
        # Пример выражения: три заглавные буквы или больше подряд, возможно с точками
        abbreviation_pattern = re.compile(r'\b[А-Я]{3,}\.?\b')
        matches = re.findall(abbreviation_pattern, content)

        # Добавляем найденные аббревиатуры в множество для уникальности
        abbreviations.update(matches)

    return abbreviations

# Пример использования
file_path = 'text.txt'  # Замените на путь к вашему файлу
result = extract_abbreviations(file_path)

print("Найденные аббревиатуры:")
for abbreviation in result:
    print(abbreviation)
