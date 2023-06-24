import unittest
import numpy as np
import os
from model_utils import load_model_from_checkpoint, load_hyperparameters_from_json, dense_model

class TestModelLoading(unittest.TestCase):
    def setUp(self):
        self.test_trial_dir = "./my_dir/Stock_Trading_dense/trial_0017"
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

        # Check if the predictions from the new model and the loaded model are significantly different
        self.assertGreater(np.mean(np.abs(new_predictions - predictions)), 0.05)
        print("Test passed: Model's predictions changed significantly after loading weights.")
        self.test_passed += 1


        print(f"\n\nSuccess! All {self.test_passed} tests passed!")


if __name__ == "__main__":
    unittest.main()
