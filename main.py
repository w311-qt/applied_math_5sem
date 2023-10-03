import numpy as np
from scipy.optimize import linprog
import json


# Функция для чтения данных из JSON файла
def read_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


# Функция для преобразования задачи в каноническую форму
def to_canonical_form(problem):
    f = np.array(problem['f'])
    constraints = problem['constraints']

    A_eq = []  # Матрица коэффициентов равенства
    A_ub = []  # Матрица коэффициентов неравенства
    b_eq = []  # Вектор правых частей равенства
    b_ub = []  # Вектор правых частей неравенства
    bounds = []  # Ограничения на переменные

    for constraint in constraints:
        coef = np.array(constraint['coefs'])
        b = constraint['b']

        if constraint['type'] == 'eq':
            A_eq.append(coef)
            b_eq.append(b)
        elif constraint['type'] == 'lte':
            A_ub.append(coef)
            b_ub.append(b)
        elif constraint['type'] == 'gte':
            A_ub.append(-coef)  # Преобразуем неравенство в противоположное
            b_ub.append(-b)

    return f, A_eq, b_eq, A_ub, b_ub


# Функция для решения задачи линейного программирования
def solve_linear_programming(f, A_eq, b_eq, A_ub, b_ub, goal):
    c = np.array(f)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    if goal == 'max':
        c = -c  # Меняем знак целевой функции для максимизации

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub)

    if result.success:
        return result.x, -result.fun if goal == 'max' else result.fun
    else:
        return None, None


# Основная часть программы
if __name__ == "__main__":
    # Чтение данных из JSON файла
    data = read_json('lp_problem.json')

    # Преобразование задачи в каноническую форму
    f, A_eq, b_eq, A_ub, b_ub = to_canonical_form(data)

    # Решение задачи линейного программирования
    solution, optimal_value = solve_linear_programming(f, A_eq, b_eq, A_ub, b_ub, data['goal'])

    if solution is not None:
        print("Оптимальное решение:", solution)
        print("Значение целевой функции:", optimal_value)
    else:
        print("Задача не имеет решения или имеет бесконечное количество решений.")
