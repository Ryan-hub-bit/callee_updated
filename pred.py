import pickle
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from SiameseModel import ContrastiveClassifier
import numpy as np
from glob import glob
from argparse import ArgumentParser
import os
import json



class AICTPairWithPreGenEmbDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        print('Loading dataset...')
        self.emb_files = []
        self.load_data()

    def __getitem__(self, idx): # per callsite
        caller_embs = []
        callee_embs = []
        with open(self.emb_files[idx], 'rb') as f:
            call_pairs = pickle.load(f)
            for caller_sig, caller_emb, callee_sig, callee_emb in tqdm(call_pairs):
                caller_embs.append(caller_emb)
                callee_embs.append(callee_emb)
        print(self.emb_files[idx])
        return self.emb_files[idx], np.array(caller_embs), np.array(callee_embs)

    def __len__(self):
        return len(self.emb_files)

    def load_data(self):
        for slice_file in tqdm(glob('{}/*.pkl'.format(self.dataset_path))):
            self.emb_files.append(slice_file)


if torch.cuda.is_available():
    dev=torch.device('cuda')
else:
    dev=torch.device('cpu')
print(dev)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i','--emb_dir', type=str, help='embeddings dir', nargs='?', default='./aict-embeddings')
    parser.add_argument('--model', type=str, help='siamese network model', nargs='?', default='./model_bce_with_pregen_emb_2.pth')

    args = parser.parse_args()


    model = ContrastiveClassifier(3,100, 256, 128, 1, 256, 1).to(dev)
    params_load = torch.load(args.model,map_location='cpu')['state_dict']
    model.load_state_dict(params_load)
            
    

    aict_loader = DataLoader(AICTPairWithPreGenEmbDataset(args.emb_dir), batch_size = 1, num_workers=0, shuffle=True)
    model.eval()
    icts = {}
    call_info_dir = "/home/isec/Documents/slice"
    json_dir = "/home/isec/Documents/binary/json"
    json_ext = ".tgcfi.json"
    callsite_folder = "/home/isec/Documents/callsite_folder2"
    cnt = 0
    ground_truth_callee = {}
    with torch.no_grad():
        for i, (binary_name, caller_embs, callee_embs) in tqdm(enumerate(aict_loader)):
            binary_name = binary_name[0]
            # print("binary_name" + binary_name)
            #0eb777313b4dff46887c5d02cbbbd802f1244b97e8b7d94b29e66c8a42f66814_1321.slice.uniq.pkl
            binary_path = binary_name.split("/")
            # print(binary_path[2])
            # 0eb777313b4dff46887c5d02cbbbd802f1244b97e8b7d94b29e66c8a42f66814_1321
            binary_prefix = binary_path[2].split(".")[0]
            # print(binary_prefix)
            caller_embs = caller_embs.to(dev)
            caller_embs = torch.squeeze(caller_embs)
            callee_embs = callee_embs.to(dev)
            callee_embs = torch.squeeze(callee_embs)
            preds = model(caller_embs, callee_embs)
            ground_truth_arr = [0 * len(preds.cpu().numpy())]
            callee = []      
            binary_file = binary_prefix + '_f.slice'
            # binary_file='1460fcdb425c4f83e5bc7682c6621f652b2faee6bd42ba72253616737f8ded34_2_f.slice'
            # print(binary_file)
            callsite = ''
            if os.path.exists(os.path.join(call_info_dir, binary_file)):
              with open(os.path.join(call_info_dir, binary_file),"r") as f:
                  lines = f.readlines()
                  i = 0
                  for line in lines:
                      if i == 0:
                        #   print(line)
                          callsite = line.split(":")[1]
                          i += 1
                          continue
                      callee.append(line.rstrip("\n"))
                      
            # print("numpy:" + str(len(preds.cpu().numpy())))
            # print("callee: " +str(len(callee)))
                    #   print(line.rstrip("\n"))
            #   print(callee)
            # print(callsite)
            # binary_prefix = "1460fcdb425c4f83e5bc7682c6621f652b2faee6bd42ba72253616737f8ded34"
#             json_f= open(os.path.join(json_dir,
#  "1460fcdb425c4f83e5bc7682c6621f652b2faee6bd42ba72253616737f8ded34.tgcfi.json"))
            json_f= open(os.path.join(json_dir,binary_prefix.split("_")[0] + json_ext))
            data = json.load(json_f)
            # with open (os.path.join(callsite_folder,binary_file),"w") as f:
            #     f.write("callsite:" + callsite)
            target = data["tg_targets"]
            for ss in target:
                # with open (os.path.join(callsite_folder,binary_file),"a") as f:
                #     f.write(ss+"\n")
                arr = ss.split("@")
                # print(arr[0])
                curCallsite = arr[0].split(" ")[-1]
                # print(curCallsite)
                # with open("/home/isec/Documents/Callee/res.txt","a") as f:
                #         f.write("callsite" + callsite + "---- Current Callsite" +curCallsite)
                if callsite.strip() == curCallsite.strip():
                    # print(callsite +"\n")
                    # print(ss)
                    cnt += 1
                    with open("/home/isec/Documents/Callee/res.txt","a") as f:
                        f.write("callsite " + callsite + "\nkey " +''.join(target[ss]) + "\n")
                        f.write("json_file: " + binary_prefix.split("_")[0] + json_ext +"\n" )
                        f.write("binary_file: " + binary_prefix +"\n" )
            #         print(f'Callsite is{callsite} : ss is {ss}')
            #         with open("/home/isec/Documents/Callee/res.txt","a") as f:
            #             f.write(callsite + ss)
                    # ground_truth_callee[callsite] = target[ss]

                    # for t in target[ss]:
                    # print("here")
                    for i in range(len(ground_truth_arr)):
                            if callee[i] in target[ss]:
                                ground_truth_arr[i] = 1
                    print(ground_truth_arr)
                    break
                    
            # print(f'Callsite {i}, preds:', preds.cpu().numpy())
    print(cnt)
