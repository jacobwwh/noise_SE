import torch
import numpy as np
import torch.nn.functional as F

def calculate_tracin_score(
        args,
        model,
        weights_paths,
        train_dataloader,
        nu_for_each_epoch,
):
    LR = nu_for_each_epoch

    indexes = []
    model.load_state_dict(torch.load(weights_paths))
    score_matrix = np.zeros(len(train_dataloader))
    for x_index, x_train, y_train in iter(train_dataloader):
        grad_sum = 0
        if x_index % 1000 == 0:
            print(x_index.item(), ' / ', len(train_dataloader))
        x_index.to(args.device)
        x_train = x_train.to(args.device)
        y_train = y_train.to(args.device)
        indexes.append(x_index.cpu().item())
        model.eval()
        loss, _, _ = model(x_train, y_train)
        loss.backward()  # back
        train_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
        grad_sum += LR * np.dot(train_grad.cpu().numpy(),
                                train_grad.cpu().numpy())
        model.zero_grad()

        score_matrix[x_index] += grad_sum

    return score_matrix, indexes