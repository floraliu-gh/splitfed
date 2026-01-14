import copy
import torch

def fedserver(client_models, nk_list, n):
    """
    FedAvg: 聚合所有 clients 的模型
    參數:
    - client_models: list of nn.Module (所有 client 的模型)
    - nk_list: list of int (每個 client 的樣本數)
    - n: int (總樣本數)
    回傳:
    - global_model: 聚合後的全局模型
    """
    global_model = copy.deepcopy(client_models[0])
    global_dict = global_model.state_dict()

    # 初始化為 0
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])

    # 加權平均
    for k, client_model in enumerate(client_models):
        client_dict = client_model.state_dict()
        weight = nk_list[k] / n
        
        for key in global_dict.keys():
            # 只對浮點數參數做加權平均
            if global_dict[key].dtype in [torch.float32, torch.float64, torch.float16]:
                global_dict[key] += weight * client_dict[key]
            else:
                # 對於整數類型 (如 BatchNorm 的 num_batches_tracked)，取最後一個 client 的值
                global_dict[key] = client_dict[key]

    global_model.load_state_dict(global_dict)
    return global_model