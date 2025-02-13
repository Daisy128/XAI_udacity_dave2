import os

from statsmodels.tsa.vector_ar import output
from tqdm import tqdm
from utils import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from functools import partial
from tensorflow.keras.models import load_model
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from alibi.explainers import IntegratedGradients

class AttentionMapGenerator:
    def __init__(self, model, focus="steering"):
        """
        Initialize the class with a model and the focus parameter.
        """
        self.model = model
        self.focus = focus

    def smooth_grad(self, x):
        saliency = Saliency(self.model,
                            model_modifier=None,
                            clone=True)  # normalize_map=False   , also GradCam_pp, faster_score_cam, IntegratedGradients
        score = saliency(partial(self.score_decrease),
                                x,
                                smooth_samples=20,
                                smooth_noise=0.20)
        return score

    def raw_smooth_grad(self, x):
        saliency = Saliency(self.model,
                            normalize_map=False,
                            model_modifier=None,
                            clone=True)  # normalize_map=False   , also GradCam_pp, faster_score_cam, IntegratedGradients
        score = saliency(partial(self.score_decrease),
                         x,
                         smooth_samples=20,
                         smooth_noise=0.20)
        return score

    def grad_cam_pp(self, x):
        gradcam = GradcamPlusPlus(self.model,
                                  model_modifier=None,
                                  clone=True)
        # Generate heatmap with GradCAM
        cam = gradcam(partial(self.score_decrease),
                      x,
                      penultimate_layer=-1)
        return cam

    def faster_score_cam(self, x):
        # Create ScoreCAM object
        scorecam = Scorecam(self.model,
                            model_modifier=None)
        # Generate heatmap with Faster-ScoreCAM
        cam = scorecam(partial(self.score_decrease),
                       x,
                       penultimate_layer=-1,
                       max_N=10)
        return cam

    def integrated_gradients(self, X, steps=10):
        ig = IntegratedGradients(self.model,
                                 n_steps=steps,
                                 method="gausslegendre")
        #predictions = self.model(X).numpy().argmax(axis=1)
        predictions = np.ones((X.shape[0])) * 3
        predictions = predictions.astype(int)
        explanation = ig.explain(X,
                                 baselines=None,
                                 target=predictions)
        attributions = explanation.attributions[0]
        # remove single-dimensional shape of the array.
        # attributions = attributions.squeeze()
        attributions = np.reshape(attributions, (-1, 28, 28))
        # only focus on positive part
        # attributions = attributions.clip(0, 1)
        attributions = np.abs(attributions)
        normalized_attributions = np.zeros(shape=attributions.shape)


        # Normalization
        for i in range(attributions.shape[0]):
            try:
                # print(f"attention map difference {np.max(attributions[i]) - np.min(attributions[i])}")
                normalized_attributions[i] = (attributions[i] - np.min(attributions[i])) / (np.max(attributions[i]) - np.min(attributions[i]))
            except ZeroDivisionError:
                print("Error: Cannot divide by zero")
                return
        # print(normalized_attributions.shape)
        return normalized_attributions


    def score_increase(self, output):
        if self.focus == "steering":
            return output[:, 0]
        elif self.focus == "throttle":
            return output[:, 1]
        elif self.focus == "both":
            return 0.5 * output[:, 0] + 0.5 * output[:, 1]
        else:
            raise ValueError("Invalid mode. Choose 'steering', 'throttle', or 'both'")

    def score_decrease(self, output):
        if self.focus == "steering":
            return -1.0 * output[:, 0]
        elif self.focus == "throttle":
            return -1.0 * output[:, 1]
        elif self.focus == "both":
            return -1.0 * (0.5 * output[:, 0] + 0.5 * output[:, 1])
        else:
            raise ValueError("Invalid mode. Choose 'steering', 'throttle', or 'both'")

    def score_maintain(self, output):
        if self.focus == "steering":
            return tf.math.abs(1.0 / (output[:, 0] + tf.keras.backend.epsilon()))
        elif self.focus == "throttle":
            return tf.math.abs(1.0 / (output[:, 1] + tf.keras.backend.epsilon()))
        elif self.focus == "both":
            return tf.math.abs(1.0 / ((0.5 * output[:, 0] + 0.5 * output[:, 1]) + tf.keras.backend.epsilon()))
        else:
            raise ValueError("Invalid mode. Choose 'steering', 'throttle', or 'both'")

