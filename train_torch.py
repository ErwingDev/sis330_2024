import os
import torchreid
import torch
import os.path as osp

from multiprocessing import freeze_support
""" from __future__ import absolute_import
from __future__ import print_function
from __future__ import division """

dataset_name = "dataset_reid"
path = "datasets/reid-data/"+dataset_name
def getListFiles(folder) :
  array = []
  folder = path+"/"+folder+"/"
  listFiles = os.listdir(folder)
  for name in listFiles :
    aux = name.split("_")
    array.append([str(folder+name), int(aux[0])-1, int(aux[1].split("c")[1])])

  return array

class NewDataset(torchreid.data.ImageDataset):
    dataset_dir = dataset_name

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        train = getListFiles("train")
        query = getListFiles("query")
        gallery = getListFiles("gallery")

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)




def init_train() :
    if dataset_name not in torchreid.data.__dict__ : 
        torchreid.data.register_image_dataset(dataset_name, NewDataset) 
    
    freeze_support()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    datamanager = torchreid.data.ImageDataManager(
        root='datasets/reid-data/',
        sources= dataset_name,
        transforms=["random_crop"],
        height=256,
        width=128,
        batch_size_test=32,
        batch_size_train=100,
        combineall=True
    )

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )
    model.to(device)
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir="datasets/reid-data/log/osnet",
        # max_epoch=100,
        max_epoch=2,
        eval_freq=50,
        print_freq=10,
        fixbase_epoch=5,
        # open_layers='classifier'
    )

    print("proceso terminado")


""" if __name__ == '__main__':
    main() """