import argparse
from multi_nonrigid.register import register

def parse_args():
    parser = argparse.ArgumentParser(description='Launch registration')
    parser.add_argument('--datadir',type=str, help='Directory of data that must be registered')
    parser.add_argument('--set', type=str, default='val', help='dataset that is registered')
    parser.add_argument('--saveimg', type=bool, default=False, help= 'if set to true moved image will be saved as nifti')
    parser.add_argument('--savewarp', type=bool, default=True, help='if set to true the nonrigid deformation will be saved as npy file')
    parser.add_argument('--savewarpinv', type=bool, default=True, help='if set to true the inverse nonrigid deformation will be saved as npy file')
    parser.add_argument('--savevisual', type=bool, default=True, help='if set to true png with visualization of results will be saved')
    parser.add_argument('--num', type=int, default=50, help='number of image to register')
    parser.add_argument('--weeknr', type=str, default='all', help='if set to all all data is registered, otherwise only of the chosen week')
    parser.add_argument('--atlasdir', type=str, help='directory with all atlas files')
    parser.add_argument('--slurmid', type=str, help='slurm id of the job')
    parser.add_argument('--experimentid', type=str, help='experiment id')
    parser.add_argument('--weightfilenum', type=str, help='the number of the weight file to load used for registration')
    parser.add_argument('--enc', type=int, default=16, help='filters of encoder used for training') 
    parser.add_argument('--dec', type=int, default=2, help='number of layers at full resolution')
    parser.add_argument('--diffeo', type=bool, default=False, help='if true diffeomorhpic model is trained')
    parser.add_argument('--intsteps', type=int, default=7, help='number of integration steps for diffeomorphic model')
    parser.add_argument('--m', type=int, default=1, help='number of atlases used for optimization')
    parser.add_argument('--outputdir', type=str, help='directory to save output')
    parser.add_argument('--atlaslist', type=str, nargs='+', default='00002', help='list of id of used atlases')
    parser.add_argument('--test', type=bool, default=False, help='variable to enable testing')
    parser.add_argument('--rerun', type=bool, default=False, help='Indicates if new slurmid is used to do registration after training')
    parser.add_argument('--start', type=int, default=0, help='Where in the list of images start with registering')
    args = parser.parse_args()
    return args

def main(args):
    if args.rerun == False:
        save_dir = args.outputdir + '/model_out_' + args.slurmid+args.experimentid + '/warp_' + args.set
        loaded_model = args.outputdir + '/model_out_'+args.slurmid + args.experimentid + '/' + args.weightfilenum + '.h5'
    else:
        save_dir = args.outputdir + '/warp_' + args.set
        loaded_model = args.outputdir + '/' + args.weightfilenum + '.h5'

    data_dir = args.datadir + '/' + args.set

    enc = [args.enc, 2*args.enc, 2*args.enc ,2*args.enc]
    if args.dec == 2:
        dec = [2*args.enc,2*args.enc, 2*args.enc, 2*args.enc, 2*args.enc, args.enc, args.enc]
    else:
        dec = [2*args.enc, 2*args.enc, 2*args.enc, 2*args.enc, 2*args.enc]

    if args.test == True:
        print(args)
        print('data_dir: ' + data_dir)
        print('save_dir: ' + save_dir)
        print(enc)
        print(dec)
    else:
        register(args.saveimg, args.savewarp, args.savewarpinv, args.savevisual, args.num, args.weeknr, data_dir, args.atlasdir, save_dir, loaded_model, args.atlaslist, args.m, enc, dec, args.diffeo, args.intsteps, args.start)


if __name__ == '__main__':
    args = parse_args()
    main(args)

  

