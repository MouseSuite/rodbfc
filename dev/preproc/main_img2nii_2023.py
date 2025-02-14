import nilearn as nl
import os
import nilearn.image as ni
from nilearn.image.image import load_img
from nilearn.plotting import plot_anat, plot_img
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np

from glob import glob 

subdir_uncorr = '/deneb_disk/rodent_bfc_data4ML/BiasFieldCorrection/BiasFieldCorrection'
subdir_corr = '/deneb_disk/rodent_bfc_data4ML/BFCorrected'
outdir = '/deneb_disk/rodent_bfc_data4ML/data4ML2'

flist = [os.path.basename(x) for x in glob(subdir_uncorr + '/*.hdr')]

for s in flist:
    subid = s[:-4]
    subfile = os.path.join(subdir_uncorr, subid + '.hdr')
    img = nb.load(subfile)
    outfname = os.path.join(outdir,subid + '_uncorr.nii.gz')
    nb.save(img,outfname)

    os.system("fslorient -deleteorient " + outfname)
    os.system("fslswapdim " + outfname + " -x -y z " + outfname)
    os.system("fslorient -setqform 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 " + outfname)
    os.system("fslorient -setqformcode 1 " + outfname)

    subfile_corr = os.path.join(subdir_corr, subid + '.hdr')
    img = nb.load(subfile_corr)
    
    outfname = os.path.join(outdir,subid + '_corr.nii.gz')

    nb.save(img, outfname)

    os.system("fslmaths " + outfname + " -div 2 -thr -0 " + outfname )
    os.system("fslorient -deleteorient " + outfname)
    os.system("fslswapdim " + outfname + " x z -y " + outfname)
    os.system("fslorient -setqform 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 " + outfname)
    os.system("fslorient -setqformcode 1 " + outfname)

