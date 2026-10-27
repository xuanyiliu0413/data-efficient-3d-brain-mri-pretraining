
import torch

class CoxPHLoss(torch.nn.Module):













    def __init__(self, reduction: str = 'mean', l2_lambda: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.l2_lambda = l2_lambda

    def forward(self, risk_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:










        sorted_indices = torch.argsort(time, descending=True)
        sorted_risk = risk_pred[sorted_indices]
        sorted_event = event[sorted_indices]



        max_risk = sorted_risk.max()
        exp_risk = torch.exp(sorted_risk - max_risk)
        cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
        log_risk_set = torch.log(cumsum_exp_risk + 1e-7) + max_risk


        uncensored_likelihood = sorted_risk - log_risk_set
        censored_mask = sorted_event.bool()

        if censored_mask.sum() == 0:

            return risk_pred.sum() * 0.0

        neg_log_likelihood = -uncensored_likelihood[censored_mask]

        if self.reduction == 'mean':
            cox_loss = neg_log_likelihood.mean()
        elif self.reduction == 'sum':
            cox_loss = neg_log_likelihood.sum()
        else:
            return neg_log_likelihood


        if self.l2_lambda > 0:
            l2_penalty = self.l2_lambda * risk_pred.pow(2).mean()
            cox_loss = cox_loss + l2_penalty

        return cox_loss




def concordance_index(risk_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
















    risk_pred = risk_pred.detach().cpu().float().numpy()
    time = time.detach().cpu().float().numpy()
    event = event.detach().cpu().float().numpy()

    n = len(time)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):

            if time[i] == time[j]:
                continue


            if time[i] < time[j]:
                shorter_idx, longer_idx = i, j
            else:
                shorter_idx, longer_idx = j, i


            if event[shorter_idx] == 0:
                continue


            if risk_pred[shorter_idx] > risk_pred[longer_idx]:
                concordant += 1
            elif risk_pred[shorter_idx] < risk_pred[longer_idx]:
                discordant += 1
            else:
                tied_risk += 0.5

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5

    return (concordant + tied_risk) / total


def fast_concordance_index(risk_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:





    risk_pred = risk_pred.detach().cpu().float().numpy()
    time = time.detach().cpu().float().numpy()
    event = event.detach().cpu().float().numpy()

    n = len(time)


    time_diff = time[:, None] - time[None, :]
    risk_diff = risk_pred[:, None] - risk_pred[None, :]


    valid_pairs = (event[:, None] == 1) & (time_diff < 0)


    concordant = (valid_pairs & (risk_diff > 0)).sum()


    discordant = (valid_pairs & (risk_diff < 0)).sum()


    tied = (valid_pairs & (risk_diff == 0)).sum() * 0.5

    total = concordant + discordant + tied
    if total == 0:
        return 0.5

    return float((concordant + tied) / total)
