import os
import os.path
#import json
import numpy as np
import h5py
#from pointnet2.data.data_utils import PointcloudJitter
#import torch
import open3d as o3d
#import copy


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def loadply(fn):
    with open(fn,"r") as freader:
        header=True
        vertices_count=0
        while header:
            line = freader.readline()
            str=line.strip().split(' ')
            if str[0]=='element':
                if str[1]=='vertex':
                    vertices_count=int(str[2])
            elif str[0]=='end_header':
                header=False
            #otherwise continue
        pointset=[]
        #normals=[]
        for i in range(vertices_count):
            line = freader.readline()
            numbers=line.strip().split(' ')
            pt=[]
            pt.append(float(numbers[0]))
            pt.append(float(numbers[1]))
            pt.append(float(numbers[2]))
            '''n=[]
            n.append(float(numbers[3]))
            n.append(float(numbers[4]))
            n.append(float(numbers[5]))'''
            pointset.append(pt)
            #normals.append(n)

    return np.array(pointset)#, np.array(normals)

def drawPoints(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

def generate_h5f(path='../data/8K_4'):
    filenames = os.listdir(path)
    with h5py.File('../data/test/supplement.hdf5','w') as f:
        pss = []
        fns = []
        cens = []
        scales = []
        for fn in filenames:
            full_fn = os.path.join(path,fn)
            point_set = loadply(full_fn)
            point_set, cen, m = pc_normalize(point_set)
            cens.append(cen)
            scales.append(m)
            print(full_fn+': centroid='+str(cen)+', scale='+str(m))
            pss.append(point_set)
            '''jit1 = PointcloudJitter()
            jit2 = PointcloudJitter(0.005)
            p_add_n1 = copy.copy(point_set)
            jit1(torch.from_numpy(p_add_n1)).numpy()
            #drawPoints(p_add_n1)
            pss.append(p_add_n1)
            print(full_fn + ' noise 0.01')
            p_add_n2 = copy.copy(point_set)
            jit2(torch.from_numpy(p_add_n2)).numpy()
            pss.append(p_add_n2)
            print(full_fn + ' noise 0.005')'''
            fns.append(fn.encode('utf8'))
        f['points']=pss
        f['names']=fns
        f['centroid']=np.array(cens)
        f['scale']=np.array(scales)


def scaleTransform(plypath1, plypath2):
    ps = loadply(plypath1)
    centroid = np.mean(ps, axis=0)
    ps = ps - centroid
    m = np.max(np.sqrt(np.sum(ps ** 2, axis=1)))
    ps2 = loadply(plypath2)
    # ps2[:,[1,2]]=ps2[:,[2,1]] #if exchange y,z
    ps2 = ps2 - centroid
    ps2 = ps2 / m
    np.savetxt("scale_transformed.txt", ps2)

def normPoints(pc, cen, m):
    pc = pc - cen
    pc = pc / m
    return pc

def generate_h5forP2P(filepath):
    filenames = sorted(os.listdir(filepath))
    with h5py.File('P2P_test.hdf5','w') as f:
        pss = []
        sks = []
        for fn in filenames:
            full_fn = os.path.join(filepath,fn)
            point_set = loadply(full_fn)
            pss.append(point_set)
            sks.append(point_set)
        f['skeleton'] = sks
        f['surface'] = pss
        f['names'] = [fn.encode('utf8') for fn in filenames]

if __name__ == '__main__':
    #generate_h5f('/home/hiko/Workspace/TreeData/8K')
    with h5py.File('../data/test/supplement.hdf5', 'r') as f:
        i = 0
        cen = f['centroid'][i]
        sc = f['scale'][i]
        foliage = np.loadtxt('../../../TreeData/output/SheKouTree038_foliage.txt')
        foliage = normPoints(foliage,cen,sc)
        trunk = np.loadtxt('../../../TreeData/output/SheKouTree038_trunk.txt')
        trunk = normPoints(trunk,cen,sc)
        np.savetxt('output/Tree' + str(i) +'_ori_foliage.txt', foliage, fmt='%f')
        np.savetxt('output/Tree' + str(i) + '_ori_trunk.txt', trunk, fmt='%f')
    
