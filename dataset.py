import torch 
from torch.utils.data import Dataset
import numpy as np


def get_stimuli(path, neuron_ranges, sti_idx, cue_idx, seed=1233):
    trainsize, testsize, valsize = 0.7, 0.2, 0.1
    train_input1=torch.tensor(np.load(path+f"data/spike_data_0.1_new/stimuli{sti_idx}_cue{cue_idx}_0.1.npy"))
    slices = [torch.arange(start, end) for start, end in neuron_ranges]
    train_input1 = train_input1[:, :, torch.cat(slices, dim=-1)]
    
    train_output1=torch.tensor(np.load(path+f"data/stimuli_tensor_new/Train_stimuli{sti_idx}_cue{cue_idx-1}_frames10.npy")) # torch.Size([60, 45, 3, 64, 64])
    length=train_input1.shape[0]
    train_output_copy=train_output1.repeat(80,1,1,1,1)
    train_output1=train_output_copy[:length,:,:,:,:]

    torch.manual_seed(seed)
    random_index = torch.randperm(560)[:512]
    train_input1=train_input1[random_index]
    train_output1=train_output1[random_index]
    train_output1=train_output1[:,:,:,16:48,16:48]
    
    
    train_size=trainsize*len(train_input1)
    test_size=testsize*len(train_input1)
    val_size=valsize*len(train_input1)

    train_input=train_input1[:int(train_size)]
    train_output=train_output1[:int(train_size)]
    test_input=train_input1[int(train_size):int(train_size+test_size)]
    test_output=train_output1[int(train_size):int(train_size+test_size)]
    val_input=train_input1[int(train_size+test_size):]
    val_output=train_output1[int(train_size+test_size):]
    
    return train_input,train_output,test_input,test_output,val_input,val_output
    
def processing_stimuli(path, stimuli_list, neuron_ranges, seed):

    train_input = torch.load("/data/train_input")
    train_output = torch.load("/data/train_output")
    test_input = torch.load("/data/test_input")
    test_output = torch.load("/data/test_output")
    val_input = torch.load("/data/val_input")
    val_output = torch.load("/data/val_output")
    
    train_output = train_output[:,:,:,16:48,16:48]
    test_output = test_output[:,:,:,16:48,16:48]
    val_output = val_output[:,:,:,16:48,16:48]

    return train_input,train_output,test_input,test_output,val_input,val_output

class SpikeDataset(Dataset):
    def __init__(self, neuron_ranges, stimuli_list, categ, seed):
        super().__init__()
        self.path = '/home/user/../monkctrl/'
        self.neuron_ranges = neuron_ranges
        self.stimuli_list = stimuli_list
        self.seed = seed
        
        train_input,train_output,test_input,test_output,val_input,val_output = processing_stimuli(self.path, self.stimuli_list, self.neuron_ranges, self.seed)
        if categ == 'train':
            self.input_ds = train_input
            self.output_ds = train_output.permute(0,2,1,3,4)
            self.len = train_input.size(0)
        elif categ == 'test':
            self.input_ds = test_input
            self.output_ds = test_output.permute(0,2,1,3,4)
            self.len = test_input.size(0)
        elif categ == 'validation':
            self.input_ds = val_input
            self.output_ds = val_output.permute(0,2,1,3,4)
            self.len = val_input.size(0)
    
    def __len__(self):
        
        return self.len
    
    def __getitem__(self, index):
        
        return {"input": self.input_ds[index,:,:].float(), 
                "output":self.output_ds[index,:,:,:,:].float()
                }
        
        
        
    