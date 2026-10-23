"""Official GUE BOME core; unused upstream demos and evaluation removed."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResNet18
from utils import setup_seed


def create_net(args):
    if args.dataset == "cifar10":
            return ResNet18(num_classes=10)
    if args.dataset == "cifar100":
            return ResNet18(num_classes=100)

def f(args, clsmodel, x, y):
    logits = clsmodel(x)
    losses = []
    for i in range(1, args.num_classes):
        k = (y+i) % args.num_classes
        losses.append(nn.CrossEntropyLoss(reduction='none')(logits, k))
    losses = torch.stack(losses, dim=0)
    losses = losses.max(0)[0].mean()
    return losses

def f_theta(args, clsmodel, x, y, retain_graph=False, create_graph=False):
    loss = f(args, clsmodel, x, y)
    print("clean loss:{:.4f}".format(loss.item()))
    grad = torch.autograd.grad(loss, clsmodel.parameters(),
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return grad, loss

def g(args, atkmodel, clsmodel, x, y):
    noise = atkmodel(x) * args.eps
    atkdata = torch.clamp(x + noise, 0, 1)
    logits = clsmodel(atkdata)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss

def g_theta(args, atkmodel, clsmodel, x, y, retain_graph=False, create_graph=False):
    loss = g(args, atkmodel, clsmodel, x, y)
    print("poison loss:{:.4f}".format(loss.item()))
    grad = torch.autograd.grad(loss, clsmodel.parameters(),
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return grad

def g_v(args, atkmodel, clsmodel, x, y, retain_graph=False, create_graph=False):
    loss = g(args, atkmodel, clsmodel, x, y)
    grad = torch.autograd.grad(loss, atkmodel.parameters(),
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return grad

def q_a_theta(args, atkmodel, clsmodel, tmpmodel, x, y, retain_graph=False, create_graph=False):
    loss1 = g(args, atkmodel, clsmodel, x, y)
    loss2 = g(args, atkmodel, tmpmodel, x, y)
    loss = loss1 - loss2
    print("poison loss:{:.4f}".format(loss1.item()))
    print("approximate optimal poison loss:{:.4f}".format(loss2.item()))
    grad_q_a = torch.autograd.grad(loss, atkmodel.parameters(),
                                   retain_graph=True,
                                   create_graph=create_graph)
    grad_q_theta = torch.autograd.grad(loss1, clsmodel.parameters(),
                                       retain_graph=retain_graph,
                                       create_graph=create_graph)
    return loss, grad_q_theta, grad_q_a, loss1

def bome(args, atkmodel, clsmodel, clsoptimizer, atkoptimizer, train_loader, iterations, epoch):
    clean_loss_list = []
    poison_loss_list = []
    st_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        print("Epoch:{} batch:{}".format(epoch, batch_idx))
        data, target = data.cuda(), target.cuda()
        tmpmodel = create_net(args).cuda()
        inner_opt = torch.optim.Adam(tmpmodel.parameters(), lr=args.lr_tmp)
        tmpmodel.load_state_dict(clsmodel.state_dict())
        tmpmodel.train()

        for i in range(iterations):
            loss = g(args, atkmodel, tmpmodel, data, target)
            print("iteration:{}, batch loss:{:.4f}".format(i, loss.item()))
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        # prepare gradients
        grad_f_theta, clean_loss = f_theta(args, clsmodel, data, target)
        clean_loss_list.append(clean_loss.item())
        grad_f_theta_vec = torch.cat([grad.flatten() for grad in grad_f_theta]).view(-1)
        loss, grad_q_theta, grad_q_a, poison_loss = q_a_theta(args, atkmodel, clsmodel, tmpmodel, data, target)
        poison_loss_list.append(poison_loss.item())
        grad_q_theta_vec = torch.cat([grad.flatten() for grad in grad_q_theta]).view(-1)
        grad_q_a_vec = torch.cat([grad.flatten() for grad in grad_q_a]).view(-1)

        norm_q = (grad_q_theta_vec**2).sum() + (grad_q_a_vec**2).sum()
        norm_q_theta = (grad_q_theta_vec ** 2).sum()
        # the inner product not need to compute grad_g_a_vec since f only contain theta
        vec_product = (grad_f_theta_vec * grad_q_theta_vec).sum()

        print("q:{:.4f}, inner product:{:.4f}, "
              "norm_q:{:.4f}, norm_q_theta:{:.4f}".format(loss.item(), vec_product.item(), norm_q.item(), norm_q_theta.item()))
        lmbd = F.relu((args.eta * norm_q - vec_product) / (norm_q + 1e-8))
        print("lambda:{}".format(lmbd.item()))
        atkmodel.train()
        clsmodel.train()
        atkoptimizer.zero_grad()
        for i, param in enumerate(atkmodel.parameters()):
            param.grad = lmbd * grad_q_a[i]
        atkoptimizer.step()

        clsoptimizer.zero_grad()
        for i, param in enumerate(clsmodel.parameters()):
            param.grad = grad_f_theta[i] + lmbd * grad_q_theta[i]
        clsoptimizer.step()
    end_time = time.time()
    clean_loss = sum(clean_loss_list) / len(clean_loss_list)
    poison_loss = sum(poison_loss_list) / len(poison_loss_list)
    print("time:{}".format(end_time-st_time))
    return clean_loss, poison_loss
