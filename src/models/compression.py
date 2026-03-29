import torch
from torch import nn
import torch.nn.utils.prune as prune
import numpy as np

def compute_neuron_importance(model, data_loader, device):
    model.to(device)
    model.eval()
    model.zero_grad()

    importances = {}
    activations = {}
    gradients = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    def get_gradient(name):
        def hook(model, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook

    hooks = []
    target_layers = {}
    for name, module in model.net.named_children():
        if isinstance(module, nn.Linear) and module.out_features > 1:
            target_layers[name] = module

    for name, layer in target_layers.items():
        hooks.append(layer.register_forward_hook(get_activation(name)))
        hooks.append(layer.register_full_backward_hook(get_gradient(name)))

    try:
        inputs, targets = next(iter(data_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = model.loss_fn(outputs, targets)
        loss.backward()

        for name in target_layers:
            act = activations[name]
            grad = gradients[name]
            imp = torch.abs(act * grad).mean(dim=0)
            importances[name] = imp.cpu().numpy()
    finally:
        for h in hooks:
            h.remove()

    return importances

def prune_neurons_by_xai(model, scores, pruning_percentage=0.3):
    for name, module in model.net.named_children():
        if name in scores:
            score = scores[name]
            n_prune = int(len(score) * pruning_percentage)
            prune_indices = np.argsort(score)[:n_prune]

            weight_mask = torch.ones_like(module.weight)
            bias_mask = torch.ones_like(module.bias)

            for idx in prune_indices:
                weight_mask[idx, :] = 0
                bias_mask[idx] = 0

            prune.custom_from_mask(module, name='weight', mask=weight_mask)
            prune.custom_from_mask(module, name='bias', mask=bias_mask)
    return model

def compress_model_physically(pruned_model, original_input_dim):
    pruned_model.cpu()
    new_layers = []
    last_kept_indices = torch.arange(original_input_dim)

    for name, layer in pruned_model.net.named_children():
        if isinstance(layer, nn.Linear):
            is_output = (layer.out_features == 1)

            if prune.is_pruned(layer):
                prune.remove(layer, 'weight')
                if layer.bias is not None:
                    try: prune.remove(layer, 'bias')
                    except: pass

            if is_output:
                current_kept = torch.arange(layer.out_features)
            else:
                row_sums = torch.sum(torch.abs(layer.weight), dim=1)
                current_kept = torch.nonzero(row_sums).squeeze()
                if current_kept.numel() == 0: current_kept = torch.tensor([0])
                if current_kept.dim() == 0: current_kept = current_kept.unsqueeze(0)

            new_layer = nn.Linear(len(last_kept_indices), len(current_kept))
            with torch.no_grad():
                subset_weights = layer.weight[current_kept][:, last_kept_indices]
                new_layer.weight.copy_(subset_weights)
                if layer.bias is not None:
                    new_layer.bias.copy_(layer.bias[current_kept])

            new_layers.append(new_layer)
            if not is_output:
                last_kept_indices = current_kept

        elif isinstance(layer, nn.BatchNorm1d):
            new_bn = nn.BatchNorm1d(len(last_kept_indices))
            with torch.no_grad():
                new_bn.weight.copy_(layer.weight[last_kept_indices])
                new_bn.bias.copy_(layer.bias[last_kept_indices])
                new_bn.running_mean.copy_(layer.running_mean[last_kept_indices])
                new_bn.running_var.copy_(layer.running_var[last_kept_indices])
            new_layers.append(new_bn)
        else:
            new_layers.append(layer)

    return nn.Sequential(*new_layers)

def apply_dynamic_quantization(model):
    model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model