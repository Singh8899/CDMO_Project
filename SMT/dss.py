
import threading
import time
class RaisingThread(threading.Thread):
  def run(self):
    self._exc = None
    try:
      super().run()
    except Exception as e:
      self._exc = e

  def join(self, timeout=None):
    super().join(timeout=timeout)
    if self._exc:
      raise self._exc
def foo():
  time.sleep(2)
  print('hi, from foo!')
  raise Exception('exception from foo')

t = RaisingThread(target=foo)
t.start()
try:
  t.join()
except Exception as e:
  print(e)