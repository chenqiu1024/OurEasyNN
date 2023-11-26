import traceback
import multiprocessing

def func(testfunc):
    exit(0 if testfunc() else 1)

def grade_all(name, begin, end, funcs):
    grade = 0
    for q in range(begin, end):
        try:
            p = multiprocessing.Process(target = func, args=(funcs["grade_Q%d" % q], ))
            print(type(p))
            p.start()
            p.join(300)
            if p.is_alive():
                p.terminate()
                print("============Q%d timeout!============\n" % q, flush = True)
            elif p.exitcode == 0:
                print("============Q%d passed!============\n" % q, flush = True)
                grade += 1
            else:
                print("============Q%d failed!============\n" % q, flush = True)
        except Exception as e:
            print("============Q%d raised an Exception!============" % q, flush = True)
            print(traceback.format_exc())
    print("Local Testing: %d functions passed" % grade)
    
    print("*************************************************************")
    print("* You may receive 0 points unless your code tests correctly *")
    print("* in CI System. Please commit and push your code to start.  *")
    print("*************************************************************")
