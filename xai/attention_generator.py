import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import preprocess
from tensorflow.keras.models import load_model
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from alibi.explainers import IntegratedGradients
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear


class AttentionMapGenerator:
    def __init__(self, model, focus: str = "steer"):
        """
        Initialize the class with a model and the focus parameter.
        """
        self.model = model
        self.model_modifier = ReplaceToLinear()
        if focus == "steer":
            self.focus = 0
        elif focus == "throttle":
            self.focus = 1
        else:
            raise ValueError("Invalid focus parameter. Please choose 'steer' or 'throttle'.")

    def smooth_grad(self, x):
        saliency = Saliency(self.model,
                            model_modifier=self.model_modifier,
                            clone=True)
        score_function = lambda output: output[:, self.focus]
        heatmap, prediction = saliency(score_function,
                           x,
                           smooth_samples=20,
                           smooth_noise=0.20,)
        return heatmap, prediction

    def grad_cam_pp(self, x):
        gradcam = GradcamPlusPlus(self.model,
                                  model_modifier=None)
        score_function = lambda output: output[:, self.focus]
        cam, prediction = gradcam(score_function,
                                  x,
                                  penultimate_layer=-1)
        return cam, prediction

    def faster_score_cam(self, x):
        # Create ScoreCAM object
        scorecam = Scorecam(self.model,
                            model_modifier=self.model_modifier)

        score_function = lambda output: output[:, self.focus]
        # Generate heatmap with Faster-ScoreCAM
        cam, prediction = scorecam(score_function,
                       x,
                       penultimate_layer=-1,
                       max_N=10)
        return cam, prediction

    def integrated_gradients(self, x, steps=10):
        """
        Generates normalized attribution maps for input data using the Integrated Gradients method.

        Parameters:
            X: ndarray  Input data for which attributions are to be computed.
            steps: Number of steps for the integration calculation. Controls the number
                       of steps along the path from the baseline to the input. Defaults to 10.

        Returns: ndarray
                Normalized attribution maps with the same shape as input data, focusing on the
                positive part of attributions,scaled to a range of [0, 1].
        """
        ig = IntegratedGradients(self.model,
                                 n_steps=steps,
                                 method="gausslegendre")

        x = np.expand_dims(x, axis=0)
        # prediction = self.model.predict(x, verbose=0)
        prediction = None # to save memory
        explanation = ig.explain(x,
                                 baselines=0,
                                 target=0)
        attributions = np.abs(explanation.attributions[0])
        # attributions = explanation.attributions[0]
        # attributions = attributions.clip(0, 1)

        # normalization
        try:
            normalized_attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions))
        except ZeroDivisionError:
            print("Error: Cannot divide by zero")
            return

        return normalized_attributions, prediction

    def raw_smooth_grad(self, x):
        saliency = Saliency(self.model,
                            model_modifier=self.model_modifier,
                            clone=True)
        score_function = lambda output: output[:, self.focus]
        heatmap, prediction = saliency(score_function,
                           x,
                           normalize_map=False,
                           smooth_samples=20,
                           smooth_noise=0.20)
        return heatmap, prediction

    def raw_grad_cam_pp(self, x):
        gradcam = GradcamPlusPlus(self.model,
                                  model_modifier=None)
        score_function = lambda output: output[:, self.focus]
        cam, prediction = gradcam(score_function,
                                  x,
                                  penultimate_layer=-1,
                                  normalize_cam=False)
        return cam, prediction

    def raw_faster_score_cam(self, x):
        # Create ScoreCAM object
        scorecam = Scorecam(self.model,
                            model_modifier=self.model_modifier)

        score_function = lambda output: output[:, self.focus]
        # Generate heatmap with Faster-ScoreCAM
        cam, prediction = scorecam(score_function,
                                   x,
                                   penultimate_layer=-1,
                                   max_N=10,
                                   normalize_cam=False)
        return cam, prediction

    def raw_integrated_gradients(self, x, steps=10):
        """
        Generates normalized attribution maps for input data using the Integrated Gradients method.

        Parameters:
            X: ndarray  Input data for which attributions are to be computed.
            steps: Number of steps for the integration calculation. Controls the number
                       of steps along the path from the baseline to the input. Defaults to 10.

        Returns: ndarray
                Normalized attribution maps with the same shape as input data, focusing on the
                positive part of attributions,scaled to a range of [0, 1].
        """
        ig = IntegratedGradients(self.model,
                                 n_steps=steps,
                                 method="gausslegendre")

        x = np.expand_dims(x, axis=0)
        # prediction = self.model.predict(x, verbose=0)
        prediction = None # only re-check, None to save memory
        explanation = ig.explain(x,
                                 baselines=0,
                                 target=0)
        attributions = np.abs(explanation.attributions[0])

        return attributions, prediction

    def score_decrease(self, output):
        return -1.0 * output[:, self.focus]

    def score_maintain(self, output):
        return tf.math.abs(1.0 / (output[:, self.focus] + tf.keras.backend.epsilon()))

if __name__ == '__main__':

    model = load_model("/home/jiaqq/Documents/ThirdEye-II/model/ckpts/ads/track1-steer-throttle.h5")
    img = Image.open("/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/lake/lake_static_smoke_filter_scale3_log/image_logs/computed_segmentation_lake_sun/computed_1.png")
    image_resize, image_nor = preprocess(img)

    generator = AttentionMapGenerator(model, focus="steer")
    score, prediction = generator.integrated_gradients(image_nor)
    heatmap = np.squeeze(score)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)  # dpi: high-resolution output
    ax.imshow(image_resize)
    ax.imshow(heatmap, cmap='jet', alpha=0.6)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除 padding
    plt.show()

    # plt.imshow(np.squeeze(img))
    # plt.imshow(heatmap[0], cmap="jet")  # 叠加到原图上
    # plt.axis("off")
    # plt.show()

