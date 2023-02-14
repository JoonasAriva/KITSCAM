import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        n, c, d, h, w = input.size()  # c to d b to c

        # predication on raw input
        logit = self.model_arch(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit, dim=1)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            score = score.cuda()
            logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value'][0]
        # b, k, u, v = activations.size()
        k, dpt, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, d, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(torch.unsqueeze(activations[i, :, :, :], 0), 0)
                saliency_map = F.interpolate(saliency_map, size=(d, h, w), mode='trilinear', align_corners=False)

                if saliency_map.max() == saliency_map.min():
                    continue

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                output = self.model_arch(input * norm_saliency_map)
                output = F.softmax(output, dim=1)
                score = output[0][predicted_class]

                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(
            score_saliency_map_max - score_saliency_map_min).data
        # added predicted class returning
        return score_saliency_map, predicted_class

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
