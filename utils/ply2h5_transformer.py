import os
import os.path
import json
import numpy as np
import h5py

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    #primitive[:,0:3] = primitive[:,0:3] - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    #primitive[:, 0:3] = primitive[:, 0:3] / m
    #primitive[:,6] = primitive[:,6] / m
    return pc#,primitive

def loadply(fn):
    with open(fn,"r") as freader:
        header=True
        vertices_count=0
        primitives_count=0
        while header:
            line = freader.readline()
            str=line.strip().split(' ')
            if str[0]=='element':
                if str[1]=='vertex':
                    vertices_count=int(str[2])
                elif str[1]=='primitive':
                    primitives_count=int(str[2])
            elif str[0]=='end_header':
                header=False
            #otherwise continue
        pointset=[]
        normals=[]
        isfork=[]
        primitive_id=[]
        for i in range(vertices_count):
            line = freader.readline()
            numbers=line.strip().split(' ')
            pt=[]
            pt.append(float(numbers[0]))
            pt.append(float(numbers[1]))
            pt.append(float(numbers[2]))
            n=[]
            n.append(float(numbers[3]))
            n.append(float(numbers[4]))
            n.append(float(numbers[5]))
            pointset.append(pt)
            normals.append(n)
            isfork.append(int(numbers[6]))
            primitive_id.append(int(numbers[7]))
        '''primitives=[]
        for i in range(primitives_count):
            line = freader.readline()
            numbers = line.strip().split(' ')
            pr=[]
            for j in range(len(numbers)):
                pr.append(float(numbers[j]))
            primitives.append(pr)'''
    return np.array(pointset), np.array(normals), np.array(isfork), np.array(primitive_id)#, np.array(primitives)

class DatasetLoader():
    def __init__(self, root, npoints = 2048, split='train', normalize=True, with_normal=True):
        self.npoints = npoints
        self.root = root
        self.normalize = normalize
        self.with_normal = with_normal

        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([d for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([d for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([d for d in json.load(f)])

        dir_point = os.path.join(self.root, "pcd")
        fns = sorted(os.listdir(dir_point))

        if split == 'trainval':
            fns = [fn for fn in fns if ((fn in train_ids) or (fn in val_ids))]
        elif split == 'train':
            fns = [fn for fn in fns if fn in train_ids]
        elif split == 'val':
            fns = [fn for fn in fns if fn in val_ids]
        elif split == 'test':
            fns = [fn for fn in fns if fn in test_ids]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        self.datapath = []
        self.fns=fns
        for fn in fns:
            self.datapath.append(os.path.join(dir_point, fn))

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 0

    def __getitem__(self, index):
        if index in self.cache:
            point_set,normal,isfork  = self.cache[index]
        else:
            fn = self.datapath[index]
            point_set,normal,isfork,prs = loadply(fn)
            if self.normalize:
                point_set = pc_normalize(point_set)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set,normal,isfork)

        #choice = np.random.choice(len(isfork), self.npoints, replace=True)
        # resample
        #point_set = point_set[choice, :]
        #normal = normal[choice, :]
        #isfork = isfork[choice]
        #primitive_id = primitive_id[choice]
        #labels = primitive[primitive_id]
        #labels=labels[:,6]
        if self.with_normal:
            return point_set,normal,isfork,prs
        else:
            return point_set,isfork,prs

    def __len__(self):
        return len(self.datapath)

    def getfns(self):
        return  self.fns

def get_skeleton(point_set, labels, normal):
    return point_set - np.expand_dims(labels,-1) * normal

#def h5fileWriter(skeleton,surface,fns):

def generate_dataset_h5(split,npoints):
    rootdir = '/media/chao/DATA/Synthetic_Data/5-trainData_20210510'
    d = DatasetLoader(root=rootdir, split=split, npoints=npoints)
    point_sets = []
    normals=[]
    isforks=[]
    primitives=[]
    fns = d.getfns()
    fns = [str(fn[:-4]).encode('utf8') for fn in fns]
    for i in range(len(d)):
        point_set, normal, labels, pr = d[i]
        point_sets.append(point_set)
        normals.append(normal)
        isforks.append(labels)
        primitives.append(pr)
        #skeletons.append(get_skeleton(point_set, labels, normal))

    with h5py.File(os.path.join(rootdir,'tree_'+split+'.hdf5'), 'w') as f:
        #f['skeleton'] = skeletons
        #f['surface'] = point_sets
        #f['names'] = fns
        f['points']=point_sets
        f['normals']=normals
        f['isforks']=isforks
        f['primitive_id']=primitives
        f['names'] = fns


if __name__ == '__main__':
    #point_set,normal,isfork,primitive_id,primitive = loadply('./test.ply')
    generate_dataset_h5('train',npoints=8000)
    generate_dataset_h5('val', npoints=8000)
    generate_dataset_h5('test',npoints=8000)
