import math

import numpy as np
from matplotlib import pyplot


def numerical_gradient(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / 2 / h


def second_numerical_gradient(f, x):
    h = 1e-4
    return (f(x + h) + f(x - h) - 2 * f(x)) / (h * h)


def main():
    h = 1e-3
    discrete = False
    div = False

    def f(x):
        try:
            y = x*x - 1
        except TypeError:
            pass
        return y

    arange = (-2, 2)

    show_graph_div(f, arange[0], arange[1], arange)
    # draw_iterated(f, arange[0], arange[1], arange, 3)
    # return

    # compare_method(f, 1.1, repeat=7)

    # 0.002
    if discrete:
        stepsize = 0.25
        width = 0.2
    else:
        stepsize = 0.002
        width = stepsize
    arange = np.arange(arange[0], arange[1], stepsize)

    results = []
    for i in arange:
        results.append(newton_method(f, float(i), div=div))

    solutions = []
    fails = []
    for i, result in enumerate(results):
        if result[1]:
            print(f'Failed to get solution where x={arange[i]}')
            fails.append((arange[i], result[0], result[2]))
            continue
        fl = False
        for group in solutions:
            if abs(group[0][2] - result[2]) < h:
                group.append((arange[i], result[0], result[2]))
                fl = True
                break
        if not fl:
            group = [(arange[i], result[0], result[2])]
            solutions.append(group)
    # print(solutions)
    # print(fails)
    draw_result(solutions, width=width, fixedticks=discrete, oneplot=False)
    # draw_result(solutions, width=width, xlim=(arange[0], arange[-1]))


def compare_method(f, x0, ylim=None, repeat=100, scatter=False):
    x_range = np.arange(0, repeat + 1, 1)
    newton = newton_method(f, x0, div=True, h=-1, returnxn=True, repeat=repeat)
    gdr1 = gradient_descent(f, x0, rate=0.1, repeat=repeat)
    gdr2 = gradient_descent(f, x0, rate=0.05, repeat=repeat)
    gdr10 = gradient_descent(f, x0, rate=0.01, repeat=repeat)

    pyplot.figure(figsize=(24, 6))
    if ylim:
        pyplot.ylim(ylim)

    if scatter:
        s1 = pyplot.scatter(x_range, newton, s=20)
        s2 = pyplot.scatter(x_range, gdr1, s=20)
        s3 = pyplot.scatter(x_range, gdr2, s=20)
        s4 = pyplot.scatter(x_range, gdr10, s=20)
    else:
        s1 = pyplot.plot(x_range, newton)
        s2 = pyplot.plot(x_range, gdr1)
        s3 = pyplot.plot(x_range, gdr2)
        s4 = pyplot.plot(x_range, gdr10)

    pyplot.legend(labels=('Newton Method', 'Gradient Descent (r=0.1)', 'Gradient Descent (r=0.05)',
                          'Gradient Descent (r=0.01)'))
    pyplot.show()


def draw_iterated(f, xmin, xmax, ylim=(-10, 10), n=2):
    def iterated(t):
        return t - f(t) / numerical_gradient(f, t)

    pyplot.figure(figsize=(24, 6))
    pyplot.subplot(1, 3, 1)
    pyplot.title('Iterated n=1')

    pyplot.ylim(ylim)
    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')

    x = np.arange(xmin, xmax, 0.0001)

    y = iterated(x)
    pyplot.plot(x, x, color='black', alpha=0.5)
    pyplot.scatter(x, y, s=3)

    pyplot.subplot(1, 3, 2)
    pyplot.title('derivative of Iterated n=1')

    pyplot.ylim(ylim)
    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')
    pyplot.axhline(y=1, color='black', alpha=0.5)

    pyplot.scatter(x, numerical_gradient(iterated, x), s=3)

    pyplot.subplot(1, 3, 3)
    pyplot.title(f'Iterated n={n}')

    pyplot.ylim(ylim)
    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')

    for _ in range(n - 2):
        y = iterated(y)
    pyplot.scatter(x, y, s=3)
    # pyplot.show()


def draw_result(solutions, fixedticks=False, width=0.8, xlim=None, oneplot=False):
    solutions.sort(key=lambda x: x[0][2])
    pyplot.figure(figsize=(24, 5))
    color = ['lightcoral', 'orange', 'gold', 'palegreen', 'turquoise', 'deepskyblue', 'royalblue', 'darkorchid',
             'silver', 'firebrick', 'tan', 'khaki', 'springgreen', 'cyan', 'steelblue', 'indigo', 'violet', 'pink']
    data = []
    legend = []
    for i, solution in enumerate(solutions):
        if not oneplot:
            pyplot.subplot(1, len(solutions), i + 1)
            pyplot.title('x to %0.4f' % solution[0][2])
        x = [x[0] for x in solution]
        xrange = ['%0.2f' % e for e in x]
        y = [x[1] for x in solution]
        b = pyplot.bar(x, y, color=color[i % len(color)], width=width)

        if oneplot:
            data.append(b)
            legend.append('%0.4f' % solution[0][2])
        pyplot.xlabel('x')
        pyplot.ylabel('length of {x_n}')

        if xlim:
            pyplot.xlim(xlim)
        if fixedticks:
            pyplot.xticks(x, xrange)
            pass
    if oneplot:
        pyplot.legend(handles=data, labels=legend)
    pyplot.show()


def gradient_descent(f, x, rate=0.1, repeat=100):
    x_n = [x]
    x_next = 0
    flag = False
    for i in range(repeat):
        x_next = x - numerical_gradient(f, x) * rate
        x_n.append(x_next)
        x = x_next
    return x_n


def newton_method(f, x, div=False, h=1e-6, returnxn=False, repeat=1000):
    x_n = [x]
    x_next = 0
    flag = False
    while True:
        try:
            if div:
                x_next = x - numerical_gradient(f, x) / second_numerical_gradient(f, x)
            else:
                x_next = x - f(x) / numerical_gradient(f, x)
                if abs(x_next) > 1e8:
                    print(numerical_gradient(f, x))
                    raise ZeroDivisionError
            if isinstance(x_next, complex):
                raise ValueError('x is complex')
        except ZeroDivisionError:
            print(f'f\'(x) == 0 where x == {x}')
            flag = True
            break
        except ValueError:
            print(f'math domain error where x == {x}')
            flag = True
            break
        except OverflowError:
            print(f'x diverges where x == {x} (overflow)')
            flag = True
            break
        if abs(x - x_next) < h:
            break
        if len(x_n) > repeat:
            print(f'x vibrates where x_n == {x_n[-1]}, {x_n[-2]}, {x_n[-3]}, ...')
            flag = True
            break
        x_n.append(x_next)
        x = x_next
    if not returnxn:
        return len(x_n), flag, x_n[-1]
    else:
        return x_n


def show_graph(f, xmin, xmax):
    pyplot.figure(figsize=(24, 5))
    pyplot.subplot(1, 1, 1)
    pyplot.title('y=f(x)')

    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')

    x = np.arange(xmin, xmax, 0.1)
    y = f(x)
    pyplot.plot(x, y)

    pyplot.show()


def show_graph_div(f, xmin, xmax, ylim=None):
    pyplot.figure(figsize=(24, 5))
    pyplot.subplot(1, 3, 1)
    pyplot.title('y=f(x)')

    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')
    if ylim:
        pyplot.ylim(ylim)

    x = np.arange(xmin, xmax, 0.001)
    y = f(x)

    pyplot.scatter(x, y, s=3)

    pyplot.subplot(1, 3, 2)
    pyplot.title('y=f\'(x)')

    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')
    if ylim:
        pyplot.ylim(ylim)

    y = numerical_gradient(f, x)
    pyplot.scatter(x, y, s=3)

    pyplot.subplot(1, 3, 3)
    pyplot.title('y=f\'\'(x)')

    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')
    if ylim:
        pyplot.ylim(ylim)

    y = second_numerical_gradient(f, x)
    pyplot.scatter(x, y, s=3)
    # pyplot.show()


def show_mechanism():
    def f(x):
        y = x ** 3 - 3
        return y

    pyplot.subplot(1, 2, 1)
    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')
    pyplot.ylim(-25, 100)
    x = np.linspace(-0.5, 5, 100)
    y = f(x)
    pyplot.plot(x, y)

    x_n = 4
    pyplot.plot(x_n, f(x_n), color='black', marker='o')
    y = numerical_gradient(f, x_n) * (x - x_n) + f(x_n)
    pyplot.plot(x, y, color='red')

    x_next = x_n - f(x_n) / numerical_gradient(f, x_n)

    pyplot.plot(x_next, 0, color='black', marker='o')

    pyplot.subplot(1, 2, 2)
    pyplot.axvline(x=0, color='black')
    pyplot.axhline(y=0, color='black')
    pyplot.ylim(-25, 100)
    y = f(x)
    pyplot.plot(x, y)

    x_n = x_next
    pyplot.plot(x_n, 0, color='black', marker='o')
    pyplot.plot(x_n, f(x_n), color='black', marker='o')
    y = numerical_gradient(f, x_n) * (x - x_n) + f(x_n)
    pyplot.plot(x, y, color='red')

    x_next = x_n - f(x_n) / numerical_gradient(f, x_n)

    pyplot.plot(x_next, 0, color='black', marker='o')

    pyplot.show()


if __name__ == '__main__':
    main()
