import torch
import numpy as np
import torch.nn.functional as F

def calculate_tracin_score(
        args,
        model,
        weights_paths,
        train_dataloader,
        test_dataloader,
        nu_for_each_epoch,
):
    LR = nu_for_each_epoch

    indexes = []
    #for train_id, (_, x_train, y_train) in enumerate(train_dataloader):

    # test_re = []
    # print('weight path -1 is ', weights_paths[-1])
    # model.load_state_dict(torch.load(weights_paths[-1]))
    score_matrix = np.zeros(len(train_dataloader))
    for train_id, batch in enumerate(train_dataloader):
        grad_sum = 0

        if train_id % 1000 == 0:
            print(str(train_id), ' / ', len(train_dataloader))
        x_index = batch[0].to(args.device)

        x_train = batch[1].to(args.device)
        y_train = batch[2].to(args.device)
        indexes.append(x_index.cpu().item())
        #print('x_index is ', x_index.cpu().item())
        for weigt_path in weights_paths:
            model.load_state_dict(torch.load(weights_paths[-1]))
             model.eval()
            logit = model(x_train)  # pred
            loss = F.cross_entropy(logit, y_train.long(), reduction='mean')
            loss.backward()  # back
            train_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
            grad_sum += LR * np.dot(train_grad.cpu().numpy(),
                                train_grad.cpu().numpy())
            model.zero_grad()

            score_matrix[x_index] += grad_sum

    return score_matrix, indexes