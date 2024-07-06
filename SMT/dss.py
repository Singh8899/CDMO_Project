import signal
import time


def handle_timeout(signum, frame):
    raise TimeoutError


def my_task():
    for i in range(6):
        print(f"I am working something long running, step {i}")
        time.sleep(1)


signal.signal(signal. SIGABRT, handle_timeout)
signal.alarm(5)  # 5 seconds

try:
    my_task()
except TimeoutError:
    print("It took too long to finish the job")
finally:
    signal.alarm(0)

