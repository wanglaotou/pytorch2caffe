import torch
import sys

__all__ = ['annalysis_pytorch_layer_name']

def annalysis_pytorch_layer_name(nnSequence, isBias=False):
    layerList = []
    if isinstance(nnSequence, nn.Sequential):
        layersNum = len(nnSequence)

        for i in range(layersNum):
            curDict = {}
            if isinstance(nnSequence[i], nn.Conv2d):
                kernelSize = nnSequence[i].kernel_size[0]
                stride = nnSequence[i].stride[0]
                groups = nnSequence[i].groups
                inChannels = nnSequence[i].in_channels
                outChannels = nnSequence[i].out_channels
                padding = nnSequence[i].padding[0]
                dilation = nnSequence[i].dilation[0]
                cfg = {}
                cfg[kernelSize] = kernelSize
                cfg[stride] = stride
                cfg[groups] = groups
                cfg[inChannels] = inChannels
                cfg[outChannels] = outChannels
                cfg[padding] = padding
                cfg[dilation] = dilation
                if isBias == False:
                    curDict["conv"] = [cfg, nnSequence[i].weight.detach().numpy()]
                else:
                    curDict["conv"] = [cfg, nnSequence[i].weight.detach().numpy(), nnSequence[i].bias.detach().numpy()]

            elif isinstance(nnSequence[i], nn.Linear):
                cfg = {}
                cfg[outChannels] = nnSequence.out_features
                curDict["fcn"] = [cfg, nnSequence[i].weight.detach().numpy(), nnSequence]
                if isBias:
                    curDict["fcn"].append(nnSequence[i].bias.detach().numpy())

            elif isinstance(nnSequence[i], nn.BatchNorm2d):
                # [] = [ scale, beta, mean, var,]
                curDict["bn2d"] = [nnSequence[i].weight.detach().numpy(), nnSequence[i].bias.detach().numpy(), \
                                     nnSequence[i].running_mean.detach().numpy(), nnSequence[i].running_var.detach().numpy()]

            elif isinstance(nnSequence[i], nn.ReLU6):
                curDict["relu6"] = []

            elif isinstance(nnSequence[i], nn.ReLU):
                curDict["relu"] = []

            else:
                print("it has unsupport layers", nnSequence[i])
                sys.exit()

            layerList.append(curDict)
    elif isinstance(nnSequence, str):
        if nnSequence == "gap":
            curDict = {}
            curDict["gap"] = []
            layerList.append(curDict)

    return layerList