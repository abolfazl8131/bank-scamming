import unittest
from .main import system

class MyTestCase(unittest.TestCase):
    def test_bias(self):
        self.assertEquals(system()['bias'],'0.02','0.03')

    def test_variance(self):
        self.assertEqual(system()['variance'], '0.04')
    def test_accuracy(self):
        self.assertEquals(system()[ 'accuracy_in_real'], '0.98','0.97')
    def test_loss(self):
        self.assertEqual(system()['loss'], '0.05')




if __name__ == '__main__':
    unittest.main()
