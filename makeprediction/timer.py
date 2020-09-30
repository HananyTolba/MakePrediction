import timeit


def timer(number, repeat):
    def wrapper(func):
        runs = timeit.repeat(func, number=number, repeat=repeat)
        print("Time is {}.".format(sum(runs) / len(runs)))

    return wrapper
