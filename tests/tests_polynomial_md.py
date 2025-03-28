
import unittest

from poly import PolynomialMD


class PolynomialMDTestCase(unittest.TestCase):

    def test_get_set(self):
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]], support_start=(0, 0))
        
        self.assertEqual(p[0,0], 1)
        self.assertEqual(p[0,1], 2)
        self.assertEqual(p[0,2], 3)
        self.assertEqual(p[1,0], 4)
        self.assertEqual(p[1,1], 5)
        self.assertEqual(p[1,2], 6)
        self.assertEqual(p[2,0], 7)
        self.assertEqual(p[2,1], 8)
        self.assertEqual(p[2,2], 9)

        self.assertEqual(p[-2,7], 0)
        p[-2,7] = 15
        self.assertEqual(p[-2,7], 15)
        self.assertEqual(p[-3,5], 0)
        self.assertEqual(p[-1,7], 0)

        self.assertEqual(p[1,8], 0)
        p[1,8] = 72
        self.assertEqual(p[1,8], 72)
        
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]], support_start=(3, 2))
        self.assertEqual(p[3, 2], 1)

    def test_support(self):
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(0, 0))
        
        r1, r2 = p.support()
        self.assertEqual(r1, range(0, 2))
        self.assertEqual(r2, range(0, 3))

        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(0, 2))
        
        r1, r2 = p.support()
        self.assertEqual(r1, range(0, 2))
        self.assertEqual(r2, range(2, 5))

        p = PolynomialMD(
            [[[1, 2], [3, 4], [5, 6]],
             [[1, 2], [3, 4], [5, 6]]], support_start=(1, 4, 5)
        )
        r1, r2, r3 = p.support()
        self.assertEqual(r1, range(1, 3))
        self.assertEqual(r2, range(4, 7))
        self.assertEqual(r3, range(5, 7))

    def test_coeff_list(self):
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(0, 0))
        self.assertEqual(p.coeff_list(), [[1, 2, 3], [4, 5, 6]])

        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(2, 3))
        self.assertEqual(p.coeff_list(), [[1, 2, 3], [4, 5, 6]])

    def test_support_start(self):
        p = PolynomialMD(
            [[[1, 2], [3, 4], [5, 6]],
             [[1, 2], [3, 4], [5, 6]]], support_start=(0, 0, 0)
        )
        self.assertEqual(p.support_start, (0, 0, 0))

        p[-2, 0, 0] = 3
        self.assertEqual(p.support_start, (-2, 0, 0))

        p[-5, -3, -2] = 5
        self.assertEqual(p.support_start, (-5, -3, -2))

    def test_effective_degree(self):
        p = PolynomialMD(
            [[[1, 2], [3, 4], [5, 6]],
             [[1, 2], [3, 4], [5, 6]]], support_start=(0, 0, 0)
        )

        self.assertEqual(p.effective_degree(), (1, 2, 1))

        p[0, 0, 5] = 7
        self.assertEqual(p.effective_degree(), (1, 2, 5))

    def test_conjugate(self):
        p = PolynomialMD([[[0]]], (0, 0, 0))
        p[1, 2, 3] = 72 + 7j
        p[-5, -3, 1] = 36 - 5j

        self.assertEqual(p[1, 2, 3], 72 + 7j)
        self.assertEqual(p[-5, -3, 1], 36 - 5j)
        q = p.conjugate()
        self.assertEqual(q[-1, -2, -3], 72 - 7j)
        self.assertEqual(q[5, 3, -1], 36 + 5j)

    def test_add(self):
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(0, 0))
        
        q = PolynomialMD(
            [[7, 8, 9],
             [1, 3, 6]], support_start=(1, 1))
        
        self.assertEqual((p+q).coeff_list(),
                         [[1,   2,   3, 0],
                          [4, 5+7, 6+8, 9],
                          [0, 1,     3, 6]])
        
        self.assertEqual((p-q).coeff_list(),
                         [[1,   2,   3,  0],
                          [4, 5-7, 6-8, -9],
                          [0,  -1,  -3, -6]])
        
        self.assertEqual((q+35).coeff_list(),
                         [[35, 0, 0, 0],
                          [0,  7, 8, 9],
                          [0,  1, 3, 6]])
        

        




if __name__ == '__main__':
    unittest.main()