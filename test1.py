import threading
import matplotlib.pyplot as plt


class ThreadClass(threading.Thread):
    def run(self):
        # plt.figure()
        # plt.plot([1, 2, 3, 4, 5])
        # plt.show()
        print(1)


t = ThreadClass()
t.start()
print(2)
a = input("End")
