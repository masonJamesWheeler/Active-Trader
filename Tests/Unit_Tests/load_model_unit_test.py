import sys
import unittest
import numpy as np
import os
import matplotlib.pyplot as plt

from Data.Data import get_and_process_data

AlphaVantage_Free_Key = "A5QND05S0W7CU55E"

class TestDenseModel(unittest.TestCase):
    def setUp(self):
        self.test_trial_dir = "../my_dir/Stock_Trading_dense/trial_0016"
        self.test_passed = 0

    def test_load_model_from_checkpoint(self):
        # Call the function to be tested
        loaded_model = load_model_from_checkpoint(trial_dir=self.test_trial_dir, model_creation_func=dense_model)

        # Load the hyperparameters from the trial.json in the test trial directory
        hp = load_hyperparameters_from_json(trial_json_path=(os.path.join(self.test_trial_dir, 'trial.json')))

        # Check if the loaded model has the right structure
        self.assertEqual(len(loaded_model.layers), 6)  # 6 layers in the dense_model
        print("Test passed: loaded model has the right number of layers.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[1].units, hp['dense_1_units'])
        print("Test passed: first Dense layer has the correct number of units.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[2].rate, hp['dropout_1'])
        print("Test passed: first Dropout layer has the correct rate.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[3].units, hp['dense_2_units'])
        print("Test passed: second Dense layer has the correct number of units.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[4].rate, hp['dropout_2'])
        print("Test passed: second Dropout layer has the correct rate.")
        self.test_passed += 1

        np.testing.assert_almost_equal(float(loaded_model.optimizer.learning_rate), hp['learning_rate'], decimal=4)
        print("Test passed: Optimizer has the correct learning rate.")
        self.test_passed += 1

        # Now we will test if it makes predictions with a dummy input tensor
        dummy_input = np.random.rand(5, 30, 14)  # 5 samples
        predictions = loaded_model.predict(dummy_input)

        # Check if the output has the right shape
        self.assertEqual(predictions.shape, (5, 3))  # model should output probabilities for 3 classes
        print("Test passed: Model outputs predictions with correct shape.")
        self.test_passed += 1

        # Create a new model with the same architecture but random weights
        new_model = dense_model(hp)
        new_predictions = new_model.predict(dummy_input)

        tickers = ["AAPL"]
        interval = '1min'
        threshhold = 0.01
        window_size = 30
        years = 2
        months = 12

        X_train, Y_train, X_test, Y_test = get_and_process_data(tickers, interval, AlphaVantage_Free_Key, threshhold,
                                                            window_size, years, months)

        predictions = loaded_model.predict(X_train)
        print(predictions)

        print(f"\n\nSuccess! All {self.test_passed} tests passed!")

class TestConvModel(unittest.TestCase):
    def setUp(self):
        self.test_trial_dir = "../my_dir/Stock_Trading_conv/trial_0024"
        self.test_passed = 0

    def test_load_model_from_checkpoint(self):
        # Call the function to be tested
        loaded_model = load_model_from_checkpoint(trial_dir=self.test_trial_dir, model_creation_func=conv_model)

        # Load the hyperparameters from the trial.json in the test trial directory
        hp = load_hyperparameters_from_json(trial_json_path=(os.path.join(self.test_trial_dir, 'trial.json')))

        # Check if the loaded model has the right structure
        self.assertEqual(len(loaded_model.layers), 8)  # 8 layers in the conv_model
        print("Test passed: loaded model has the right number of layers.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[0].filters, hp['conv_1_filter'])
        print("Test passed: first Conv1D layer has the correct number of filters.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[0].kernel_size[0], hp['conv_1_kernel'])
        print("Test passed: first Conv1D layer has the correct kernel size.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[1].pool_size[0], hp['pool_1_size'])
        print("Test passed: first MaxPooling1D layer has the correct pool size.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[3].units, hp['dense_1_units'])
        print("Test passed: first Dense layer has the correct number of units.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[4].rate, hp['dropout_1'])
        print("Test passed: first Dropout layer has the correct rate.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[5].units, hp['dense_2_units'])
        print("Test passed: second Dense layer has the correct number of units.")
        self.test_passed += 1

        self.assertEqual(loaded_model.layers[6].rate, hp['dropout_2'])
        print("Test passed: second Dropout layer has the correct rate.")
        self.test_passed += 1

        np.testing.assert_almost_equal(float(loaded_model.optimizer.learning_rate), hp['learning_rate'], decimal=4)
        print("Test passed: Optimizer has the correct learning rate.")
        self.test_passed += 1

        # Now we will test if it makes predictions with a dummy input tensor
        dummy_input = np.random.rand(5, 30, 14)  # 5 samples
        predictions = loaded_model.predict(dummy_input)

        # Check if the output has the right shape
        self.assertEqual(predictions.shape, (5, 3))  # model should output probabilities for 3 classes
        print("Test passed: Model outputs predictions with correct shape.")
        self.test_passed += 1

        # Create a new model with the same architecture but random weights
        new_model = conv_model(hp)
        new_predictions = new_model.predict(dummy_input)

        # Check if the predictions from the new model and the loaded model are significantly different
        self.assertGreater(np.mean(np.abs(new_predictions - predictions)), 0.05)
        print("Test passed: Model's predictions changed significantly after loading weights.")
        self.test_passed += 1


        print(f"\n\nSuccess! All {self.test_passed} tests passed!")

if __name__ == "__main__":
    suite = unittest.TestSuite()

    # add tests to the TestSuite object
    suite.addTest(TestDenseModel("test_load_model_from_checkpoint"))
    suite.addTest(TestConvModel("test_load_model_from_checkpoint"))

    # create a TextTestRunner with verbosity=2 (for detailed results)
    runner = unittest.TextTestRunner(verbosity=2)

    # run the suite using the runner
    result = runner.run(suite)

    print("\nRan {} tests. {} passed, {} failed.".format(result.testsRun,
                                                         result.testsRun - len(result.failures) - len(result.errors),
                                                         len(result.failures)))




