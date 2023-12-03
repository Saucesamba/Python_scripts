
from math import sqrt, atan, cos, sin, pi
import matplotlib.pyplot as plt
import numpy as np

def frange(x, y, jump):
    while x < y:
        yield float(x)
        x += jump

# Данные //вставил свои данные
R1 = 8
R2 = 24
R3 = 6
L = 20*(10**-3)

# Первое задание + //поменял Rc на Rl, и записал формулы для K(w)
Rl= lambda w: (1j*w*L)

k = lambda w: (R2*R3)/(((R2+R3+Rl)*(R1+R2))-R2**2)

k = lambda w: 9/((10**-3)*40j*w+24)

# Второе задание +
АЧХ = []
ФЧХ = []
rng = list(range(0, 50000))
rng.remove(0)
for w in rng:
    this = k(w)
    АЧХ.append(sqrt((this.real)**2 + this.imag**2))
    ФЧХ.append(atan(this.imag / this.real))
plt.xlabel('Частота ω, рад/с')
plt.ylabel('Сила тока I, A')
plt.title("АЧХ")
plt.grid(True)
plt.plot(rng, АЧХ)
plt.show()
plt.xlabel('Частота ω, рад/с')
plt.ylabel('Сила тока I, A')
plt.title("ФЧХ")
plt.grid(True)
plt.plot(rng, ФЧХ, color='#039660')
plt.show()

# Третье задание +
ГОДОГРАФ_X = []
ГОДОГРАФ_Y = []
rng = list(range(1, 5000))
mx = k(50000000).real
origin = np.array([[0, 0, 0], [0, 0, 0]])
fig, ax = plt.subplots()
lnx = []
lny = []

for w in rng:
    this = k(w)
    ГОДОГРАФ_X.append(this.real)
    ГОДОГРАФ_Y.append(this.imag)

for w in [1, 10, 100, 1000, 2000]:
    this = k(w)
    lnx.append(this.real)
    lny.append(this.imag)
    plt.text(this.real * (1.02 if w < 11 else 0.98), this.imag * (1.04 if w != 1 else 0), f'{w if w != 1 else 0}', color='white', ha='center', va='center', fontsize=8, alpha=1, bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))

ГОДОГРАФ_X.append(mx)
ГОДОГРАФ_Y.append(0)
lnx.append(mx)
lny.append(0)
plt.text(mx*0.987, 0, f'∞', color='white', ha='center', va='center', fontsize=8, alpha=1, bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))
plt.plot(ГОДОГРАФ_X, ГОДОГРАФ_Y)
plt.plot(lnx, lny, marker='o', linestyle='', markersize=8)
plt.grid(True)
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.5)
ax.set_aspect('equal', adjustable='box')
plt.title("Годограф Найквиста")
plt.xlabel('RE')
plt.ylabel('IM')
plt.show()

# Четвертое задание +
w = 2 * 10**3
Амплитуда = 10

this = k(w)
print(this)

U_входное = lambda t: Амплитуда*sin(w*t)

Сдвиг_по_амплитуде = sqrt((this.real)**2 + this.imag**2)
Сдвиг_по_фазе = atan(this.imag / this.real)

Фаза_выходная = Сдвиг_по_фазе
Амплитуда_выходная = Амплитуда * Сдвиг_по_амплитуде

U_выходное = lambda t: Амплитуда_выходная*sin(w*t + Фаза_выходная)
print("Сдвиг по амплитуде: ", Сдвиг_по_амплитуде,'\n Сдвиг по фазе: ', Сдвиг_по_фазе) #оформил красивый вывод

Диаграмма_входная = []
Диаграмма_выходная = []
rng = list(frange(-0.0001, 0.001, 0.000001)) #поменял диапазон, потому что не отображались пересечения с осью Х

for t in rng:
    Диаграмма_входная.append(U_входное(t))
    Диаграмма_выходная.append(U_выходное(t))

plt.title("Временные диаграммы напряжения (увеличено)")
plt.plot(rng, Диаграмма_входная)
plt.plot(rng, Диаграмма_выходная)

plt.grid(True)

plt.ylabel('Напряжение U, В')
plt.xlabel('Время t, c')
ax = plt.gca()

plt.show()

Диаграмма_входная = []
Диаграмма_выходная = []
rng = list(frange(0, 0.01, 0.000001))

for t in rng:
    Диаграмма_входная.append(U_входное(t))
    Диаграмма_выходная.append(U_выходное(t))

plt.title("Временные диаграммы напряжения")
plt.plot(rng, Диаграмма_входная)
plt.plot(rng, Диаграмма_выходная)

plt.grid(True)

plt.ylabel('Напряжение U, В')
plt.xlabel('Время t, c')
ax = plt.gca()

plt.show()

# Пятое задание +
an = lambda n: (2/(pi*n))
an_вых = lambda n: an(n)*Сдвиг_по_амплитуде

itn = lambda n,x: an(n)*cos(abs(n)*2000*x)
out = lambda n,x: Сдвиг_по_амплитуде*an(n)*cos(abs(n)*2000*x + Фаза_выходная)

rng = []
rngx = list(frange(-0.005, 0.005, 0.000001))
ls_out = []
ls_in = []

for i in range(1, 11, 4):
    rng.extend([i, -i-2])

for x in rngx:
    sm = 1/2
    si = 1/2
    for n in rng:
        sm += out(n,x)
        si += itn(n,x)
    ls_in.append(si)
    ls_out.append(sm)


plt.plot(rngx, ls_out)
plt.plot(rngx, ls_in)

plt.grid(True)

plt.ylabel('Напряжение U, В')
plt.xlabel('Время t, c')
plt.title("Графики сумм 9 гармоник")
ax = plt.gca()
plt.show()

rng = []
rngx = list(frange(-0.005, 0.005, 0.000001))
ls_out = []
ls_in = []

for i in range(1, 1000, 4):
    rng.extend([i, -i-2])

for x in rngx:
    sm = 1/2
    si = 1/2
    for n in rng:
        sm += out(n,x)
        si += itn(n,x)
    ls_in.append(si)
    ls_out.append(sm)


plt.plot(rngx, ls_out)
plt.plot(rngx, ls_in)

plt.grid(True)

plt.ylabel('Напряжение U, В')
plt.xlabel('Время t, c')
plt.title("Графики сумм 1000 гармоник")
ax = plt.gca()
plt.show()

for n in rng[:5]:
    plt.title(str(abs(n))+"-ая гармоника")
    plt.plot(rngx, [out(n,x) for x in rngx])
    plt.plot(rngx, [itn(n,x) for x in rngx])

    plt.grid(True)

    plt.ylabel('Напряжение U, В')
    plt.xlabel('Время t, c')
    ax = plt.gca()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.show()

# Шестое задание + //ничего не менял)
АЧХ = []
rnga = list(range(0, 5000))
rnga.remove(0)
for w in rnga:
    this = k(w)
    АЧХ.append(sqrt((this.real)**2 + this.imag**2))
    ФЧХ.append(atan(this.imag / this.real))


plt.xlabel('Частота ω, рад/с')
plt.ylabel('Сила тока I, A')
plt.ylim(0, 0.8)
plt.grid(True)
plt.plot(rnga, АЧХ, color='#039660')
ax = plt.gca()
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

plt.title('АЧХ')

plt.show()

y_val = []
for n in range(1,10):
    if (n%2 == 0):
        out = r'\frac{2}{'+str(n)+'\pi}'
        plt.plot([n, n], [0,0], marker='x', color='#1f77b4', linestyle='-')
        plt.text(n, 0.07, f'${out}$', color='white', ha='center', va='center', fontsize=10, alpha=1, bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))
    else:
        res = an(n)
        out = r'\frac{2}{'+str(n)+'\pi}'
        plt.plot([n, n], [0, res], marker='$--$', color='#1f77b4', linestyle='-')
        plt.text(n, res+0.07, f'${out}$', color='white', ha='center', va='center', fontsize=14, alpha=1, bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))

plt.ylabel('$|a_n|$ входной')
plt.xlabel('n')
plt.ylim(0, 0.8)
plt.xticks(range(1,10))
plt.grid(True)
ax = plt.gca()
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

plt.title('Модуль спектра входного сигнала ')
plt.show()

y_val = []
for n in range(1,10):
    if (n%2 == 0):
        out = r'\frac{2}{'+str(n)+'\pi}'
        plt.plot([n, n], [0,0], marker='x', color='#cc7a00', linestyle='-')
        plt.text(n, 0.07, f'${out}$', color='white', ha='center', va='center', fontsize=10, alpha=1, bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))
    else:
        res = an_вых(n)
        out = r'\frac{2}{'+str(n)+'\pi}'
        plt.plot([n, n], [0, res], marker='$--$', color='#cc7a00', linestyle='-')
        plt.text(n, res+0.07, f'${out}$', color='white', ha='center', va='center', fontsize=14, alpha=1, bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))

plt.ylabel('$|a_n|$ выходной')
plt.xlabel('n')
plt.ylim(0, 0.8)
plt.xticks(range(1,10))
plt.grid(True)
ax = plt.gca()
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

plt.title('Модуль спектра выходного сигнала ')
plt.show()

# Седьмое задание +
z_вход = lambda w: (19200+32j*w)/(1500+1j*w) #написал свое z(w)

модуль = []
фаза = []
активная = []
реактивная = []
rng = list(range(1, 5000))

for w in rng:
    this = z_вход(w)
    модуль.append(sqrt((this.real)**2 + this.imag**2))
    фаза.append(atan(this.imag / this.real))
    активная.append(this.real)
    реактивная.append(this.imag)

plt.xlabel('Частота ω, рад/с')
plt.ylabel('Z, Ом')
plt.title("Модуль сопротивления")
plt.grid(True)
plt.plot(rng, модуль)
plt.show()

plt.xlabel('Частота ω, рад/с')
plt.ylabel('Z, Ом')
plt.title("Фаза сопротивления")
plt.grid(True)
plt.plot(rng, фаза, color='#cc7a00')
plt.show()

plt.xlabel('Частота ω, рад/с')
plt.ylabel('Z, Ом')
plt.title("Активная часть сопротивления")
plt.grid(True)
plt.plot(rng, активная, color='#039660')
plt.show()

plt.xlabel('Частота ω, рад/с')
plt.ylabel('Z, Ом')
plt.title("Реактивная часть сопротивления")
plt.grid(True)
plt.plot(rng, реактивная, color='#DC143C')
plt.show()

# Восьмое задание+ \\ ничего не менял
w = 2 * 10**3
Амплитуда = 1
z = z_вход(w)
U_входное = lambda t: Амплитуда*sin(w*t)
A_входное = lambda t: (Амплитуда/sqrt((z.real)**2 + z.imag**2))*sin(w*t + atan(z.imag / z.real))

Диаграмма_входная_U = []
Диаграмма_входная_A = []
rng = list(frange(0, 0.01, 0.000001))

for t in rng:
    Диаграмма_входная_U.append(U_входное(t))
    Диаграмма_входная_A.append(A_входное(t))

plt.title("Временная диаграмма напряжения")
plt.plot(rng, Диаграмма_входная_U)


plt.grid(True)

plt.ylabel('Напряжение U, В')
plt.xlabel('Время t, c')
ax = plt.gca()

plt.show()

plt.title("Временная диаграмма тока")
plt.plot(rng, Диаграмма_входная_A, color='#cc7a00')

plt.grid(True)

plt.ylabel('Сила тока I, А')
plt.xlabel('Время t, c')
ax = plt.gca()

plt.show()