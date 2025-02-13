
from utils import utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from alibi.explainers import IntegratedGradients
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear


class AttentionMapGenerator:
    def __init__(self, model, focus: int=0):
        """
        Initialize the class with a model and the focus parameter.
        """
        self.model = model
        self.model_modifier = ReplaceToLinear()
        self.focus = focus

    def smooth_grad(self, x):
        saliency = Saliency(self.model,
                            model_modifier=None,
                            clone=True)
        score_function = lambda output: output[:, self.focus]
        heatmap = saliency(score_function,
                           x,
                           smooth_samples=20,
                           smooth_noise=0.20)
        return heatmap

    def raw_smooth_grad(self, x):
        saliency = Saliency(self.model,
                            model_modifier=None,
                            clone=True)
        score_function = lambda output: output[:, self.focus]
        heatmap = saliency(score_function,
                           x,
                           normalize_map=False,
                           smooth_samples=20,
                           smooth_noise=0.20)
        return heatmap

    def grad_cam_pp(self, x):
        gradcam = GradcamPlusPlus(self.model,
                                  model_modifier=self.model_modifier,
                                  clone=True)
        # Generate heatmap with GradCAM
        score_function = lambda output: output[:, self.focus]
        cam = gradcam(score_function,
                      x,
                      penultimate_layer=-1)
        return cam

    def faster_score_cam(self, x):
        # Create ScoreCAM object
        scorecam = Scorecam(self.model,
                            model_modifier=self.model_modifier)

        score_function = lambda output: output[:, self.focus]
        # Generate heatmap with Faster-ScoreCAM
        cam = scorecam(score_function,
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

    def score_decrease(self, output):
        return -1.0 * output[:, self.focus]

    def score_maintain(self, output):
        return tf.math.abs(1.0 / (output[:, self.focus] + tf.keras.backend.epsilon()))

if __name__ == '__main__':

    model = load_model("/home/jiaqq/Documents/ThirdEye-II/model/ckpts/ads/track1-steer-throttle.h5")
    img = Image.open("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/lake/lake_static_smoke_filter_scale3_log/image_logs/135.png")
    img_array = utils.resize(np.array(img)).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Find the last convolutional layer
    # for layer in model.layers[::-1]:
    #     if isinstance(layer, tf.keras.layers.Conv2D):
    #         print(f"Using layer: {layer.name}")  # Debug
    #         target_layer = layer.name
    #         break
    saliency = Saliency(model,
                        model_modifier=ReplaceToLinear(),
                        clone=True)  # normalize_map=False, also GradCam_pp, faster_score_cam, IntegratedGradients
    score_function = lambda output: output[:, 0]
    heatmap = saliency(score_function,
                       img_array,
                       normalize_map=False,)
    # gradcam = GradcamPlusPlus(model,
    #                             model_modifier=ReplaceToLinear())
    # score_function = lambda output: output[:, 0]
    # heatmap = gradcam(score_function, img_array, penultimate_layer=-1)
    # scorecam = Scorecam(model,
    #                     model_modifier=None)
    # score_function = lambda output: output[:, 1]
    # heatmap = scorecam(score_function, img_array, penultimate_layer=-1, max_N=10)
    plt.imshow(img)
    plt.imshow(heatmap[0], cmap="jet")  # 叠加到原图上
    plt.axis("off")
    plt.show()

