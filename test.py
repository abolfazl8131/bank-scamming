import unittest
from .main import system

class MyTestCase(unittest.TestCase):
    def test_bias(self):
        self.assertEqual(system()['bias'], '0.02')
    def test_variance(self):
        self.assertEqual(system()['variance'], '0.05')
    def test_accuracy(self):
        self.assertEqual(system()[ 'accuracy_in_real'], '0.98')
    def test_loss(self):
        self.assertEqual(system()['loss'], '0.06')




if __name__ == '__main__':
    unittest.main()
