from megatron import print_rank_0


def inspect_optimizer(model, optimizer):
    id_to_name = {param.data_ptr(): name for name, param in model.named_parameters()}

    for group in optimizer.param_groups:
        print_rank_0(group['lr'])
        print_rank_0(group['weight_decay'])
        for param in group['params']:
            print_rank_0(f"{id_to_name[param.data_ptr()]}\t\t\t\t{param.shape}"
                         f"\t{param.mean().item()}\t{param.std().item()}")
        print_rank_0("")
