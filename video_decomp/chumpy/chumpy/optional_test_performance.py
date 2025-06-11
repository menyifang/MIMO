#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import unittest
import numpy as np
from functools import reduce


has_ressources = True
try:
  import resource
  
  def abstract_ressource_timer(): 
    return resource.getrusage(resource.RUSAGE_SELF)
  def abstract_ressource_counter(r1, r2):
    _r1 = r1.ru_stime + r1.ru_utime
    _r2 = r2.ru_stime + r2.ru_utime
    
    return _r2 - _r1 
except ImportError:
  has_ressources = False
  pass


if not has_ressources:
  try:
    from ctypes import *
    
    
    
    def abstract_ressource_timer():
      val = c_int64() 
      windll.Kernel32.QueryPerformanceCounter(byref(val))
      return val
    def abstract_ressource_counter(r1, r2):
      """Returns the elapsed time between r2 and r1 (r2 > r1) in milliseconds"""
      val = c_int64() 
      windll.Kernel32.QueryPerformanceFrequency(byref(val))
      
      return (1000*float(r2.value-r1.value))/val.value

  except ImportError:
    has_win32api = False


  

from . import ch



class Timer(object):

    def __enter__(self):
        self.r1 = abstract_ressource_timer()

    def __exit__(self, exception_type, exception_value, traceback):
        self.r2 = abstract_ressource_timer()

        self.elapsed = abstract_ressource_counter(self.r1, self.r2)

# def timer():
#     tm = resource.getrusage(resource.RUSAGE_SELF)
#     return tm.ru_stime + tm.ru_utime
#
# svd1

def timer(setup, go, n):
    tms = []
    for i in range(n):
        if setup is not None:
            setup()
            
        tm0 = abstract_ressource_timer()

        # if False:
        #     from body.misc.profdot import profdot
        #     profdot('go()', globals(), locals())
        #     import pdb; pdb.set_trace()

        go()
        tm1 = abstract_ressource_timer()

        tms.append(abstract_ressource_counter(tm0, tm1))
    
    #raw_input(tms)
    return np.mean(tms) # see docs for timeit, which recommend getting minimum



import timeit
class TestPerformance(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.mtx_10 = ch.array(np.random.randn(100).reshape((10,10)))
        self.mtx_1k = ch.array(np.random.randn(1000000).reshape((1000,1000)))

    def compute_binary_ratios(self, vecsize, numvecs):

        ratio = {}
        for funcname in ['add', 'subtract', 'multiply', 'divide', 'power']:
            for xp in ch, np:
                func = getattr(xp, funcname)
                vecs = [xp.random.rand(vecsize) for i in range(numvecs)]

                if xp is ch:
                    f = reduce(lambda x, y : func(x,y), vecs)
                    def go():
                        for v in vecs:
                            v.x *= -1
                        _ = f.r

                    tm_ch = timer(None, go, 10)
                else: # xp is np
                    def go():
                        for v in vecs:
                            v *= -1
                        _ = reduce(lambda x, y : func(x,y), vecs)

                    tm_np = timer(None, go, 10)

            ratio[funcname] = tm_ch / tm_np

        return ratio

    def test_binary_ratios(self):
        ratios = self.compute_binary_ratios(vecsize=5000, numvecs=100)
        tol = 1e-1
        self.assertLess(ratios['add'], 8+tol)
        self.assertLess(ratios['subtract'], 8+tol)
        self.assertLess(ratios['multiply'], 8+tol)
        self.assertLess(ratios['divide'], 4+tol)
        self.assertLess(ratios['power'], 2+tol)
        #print ratios


    def test_svd(self):
        mtx = ch.array(np.random.randn(100).reshape((10,10)))


        # Get times for svd
        from .linalg import svd
        u, s, v = svd(mtx)
        def setup():
            mtx.x = -mtx.x

        def go_r():
            _ = u.r
            _ = s.r
            _ = v.r

        def go_dr():
            _ = u.dr_wrt(mtx)
            _ = s.dr_wrt(mtx)
            _ = v.dr_wrt(mtx)

        cht_r = timer(setup, go_r, 20)
        cht_dr = timer(setup, go_dr, 1)

        # Get times for numpy svd
        def go():
            u,s,v = np.linalg.svd(mtx.x)
        npt = timer(setup = None, go = go, n = 20)

        # Compare
        #print cht_r / npt
        #print cht_dr / npt
        self.assertLess(cht_r / npt, 3.3)
        self.assertLess(cht_dr / npt, 2700)



    
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
    unittest.TextTestRunner(verbosity=2).run(suite)
