import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
sys.path.insert(0, project_root)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative, size_average=True):
        loss = self.loss(anchor, positive, negative)
        return loss
    
class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrative Loss
    """
    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        self.eps = 1e-7
        
    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)

        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        if len(positive_pairs) == 0 and len(negative_pairs) == 0:
            return embeddings.sum() * 0
        
        losses = []

        if len(positive_pairs) > 0:
            pos_dist_sq = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
            losses.append(pos_dist_sq)

        if len(negative_pairs) > 0:
            neg_dist_sq = (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(1)
            neg_dist = torch.sqrt(neg_dist_sq + self.eps)
            
            neg_loss = F.relu(self.margin - neg_dist).pow(2)
            losses.append(neg_loss)

        loss = torch.cat(losses, dim=0).mean() 
        return loss.mean()
    
class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        ap_distances = torch.sqrt(ap_distances.clamp(min=1e-7))

        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        an_distances = torch.sqrt(an_distances.clamp(min=1e-7))

        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)