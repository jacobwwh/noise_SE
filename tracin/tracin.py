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
    print('weight path -1 is ', weights_paths[-1])
    model.load_state_dict(torch.load(weights_paths[-1]))
    # for test_id, tbatch in enumerate(test_dataloader):
    #     if test_id % 100 == 0:
    #         print('Train:', test_id, '/',  len(test_dataloader))
    #         model.load_state_dict(torch.load(weights_paths[-1]))
    #     x_test = tbatch[1].to(args.device)
    #     y_test = tbatch[2].to(args.device)
    #     #model.load_state_dict(torch.load(w))  # checkpoint
    #     model.eval()
    #     logit = model(x_test)  # pred
    #     loss = F.cross_entropy(logit, y_test.long(), reduction='mean')
    #     loss.backward()  # back
    #     test_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
    #     test_re.append((y_test[0].cpu().numpy(), test_grad.cpu().numpy()))
    #     model.zero_grad()
    # print('len test re ', len(test_re))
    # train_re = []
    # model.load_state_dict(torch.load(weights_paths[-1]))
    # for train_id, batch in enumerate(train_dataloader):
    #     if train_id % 100 == 0:
    #         print('Train:', train_id, '/', len(train_dataloader))
    #         model.load_state_dict(torch.load(weights_paths[-1]))
    #     x_train = batch[1].to(args.device)
    #     y_train = batch[2].to(args.device)
    #     #]model.load_state_dict(torch.load(w))  # checkpoint
    #     model.eval()
    #     logit = model(x_train)  # pred
    #     loss = F.cross_entropy(logit, y_train.long(), reduction='mean')
    #     loss.backward()  # back
    #     train_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
    #     train_re.append((y_train[0].cpu().numpy(), train_grad.cpu().numpy()))
    #     model.zero_grad()
    # print('len train re ', len(test_re))
    #
    # for train_id, trainre in enumerate(train_re):
    #     for test_id, testre in enumerate(test_re):
    #         if trainre[0] != testre[0]:
    #             continue
    #         grad_sum = LR * np.dot(trainre[1], testre[1])
    #         score_matrix[train_id][test_id] += grad_sum
    # return score_matrix

    # for train_id, batch in enumerate(train_dataloader):
    #     x_index = batch[0].to(args.device)
    #     x_train = batch[1].to(args.device)
    #     y_train = batch[2].to(args.device)
    #     indexes.append(x_index.cpu().item())
    #     for test_id, tbatch in enumerate(test_dataloader):
    #         x_test = tbatch[1].to(args.device)
    #         y_test = tbatch[2].to(args.device)
    #         grad_sum = 0
    #         if y_test[0] != y_train[0]:
    #             continue
    #         for w in weights_paths:
    #             model.load_state_dict(torch.load(w))  # checkpoint
    #             model.eval()
    #             logit = model(x_train)  # pred
    #             loss = F.cross_entropy(logit, y_train.long(), reduction='mean')
    #             loss.backward()  # back
    #             train_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
    #
    #             model.load_state_dict(torch.load(w))  # checkpoint
    #             model.eval()
    #             logit = model(x_test)  # pred
    #             loss = F.cross_entropy(logit, y_test.long(), reduction='mean')
    #             loss.backward()  # back
    #             test_grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
    #
    #             grad_sum += LR * np.dot(train_grad.cpu().numpy(), test_grad.cpu().numpy())  # scalar mult, TracIn formula
    #
    #         score_matrix[train_id][test_id] += grad_sum
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