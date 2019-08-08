import torch


def select_device(force_cpu=False, is_head=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    ng = 0
    if not cuda:
        print('Using CPU\n')
    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        if is_head:
            for i in range(ng):
                print('Using CUDA device{} _CudaDeviceProperties(name={}, total_memory={}MB'.\
                        format(i, x[i].name, round(x[i].total_memory/c)))
            print('')
    return device, ng


if __name__ == "__main__":
    print(select_device())