import problems
import sys

def p1():
    for n in [500, 1000]:
        problems.problem_1b(n)
        problems.problem_1c(n)

def p2tuning():
    problems.problem_2_tuning()

def p2plotting():
    problems.problem_2_plot()

def p3tuning():
    problems.problem_3_tuning()

def p3acc():
    problems.problem_3_train_and_evaluate()

def p4():
    problems.problem_4()


if __name__ == '__main__':
    p = sys.argv[1:]
    if len(p) == 2:
        if int(p[0]) == 2 and p[1] == '-tuning':
            p2tuning()
        elif int(p[0]) == 3 and p[1] == '-tuning':
            p3tuning()
        elif int(p[0]) == 3 and p[1] == '-data':
            problems.problem_3_dataGenerator()

    if len(p) == 1:
        if int(p[0]) == 1:
            p1()
        elif int(p[0]) == 2:
            p2plotting()
        elif int(p[0]) == 3:
            p3acc()
        elif int(p[0]) == 4:
            p4()

