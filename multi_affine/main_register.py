import argparse
from multi_affine.register import register

def parse_args():
    parser = argparse.ArgumentParser(description='Launch registration')
    parser.add_argument('--datadir',type=str, help='Directory of data that must be registered')
    parser.add_argument('--set', type=str, default='val', help='dataset that is registered')
    parser.add_argument('--saveimg', type=bool, default=False, help= 'if set to true moved image will be saved as nifti')
    parser.add_argument('--savewarp', type=bool, default=True, help='if set to true affine transformation matrix T will be saved as npy file')
    parser.add_argument('--savevisual', type=bool, default=True, help='if set to true png with visualization of results will be saved')
    parser.add_argument('--num', type=int, default=50, help='number of image to register')
    parser.add_argument('--weeknr', type=str, default='all', help='if set to all all data is registered, otherwise only of the chosen week')
    parser.add_argument('--atlasdir', type=str, help='directory with all atlas files')
    parser.add_argument('--slurmid', type=str, help='slurm id of the job')
    parser.add_argument('--experimentid', type=str, help='experiment id')
    parser.add_argument('--weightfilenum', type=str, help='the number of the weight file to load used for registration')
    parser.add_argument('--enc', type=int, default=64, help='filters of encoder used for training')
    parser.add_argument('--outputdir', type=str, help='directory to save output')
    parser.add_argument('--test', type=bool, default=False, help='variable to enable testing')
    parser.add_argument('--rerun', type=bool, default=False, help='Indicates if new slurmid is used to do registration after training')
    args = parser.parse_args()
    return args

def main(args):
    if args.rerun == False:
        save_dir = args.outputdir + '/model_out_' + args.slurmid + args.experimentid +'/warp_' + args.set
        loaded_model = args.outputdir + '/model_out_' + args.slurmid +args.experimentid +'/' + args.weightfilenum +'.h5'
    else:
        save_dir = args.outputdir + '/warp_' + args.set
        loaded_model = args.outputdir +'/' + args.weightfilenum + '.h5' 
    
    data_dir = args.datadir + '/' + args.set
    enc_filters = [args.enc, 2*args.enc, 2*args.enc, 2*args.enc]

    if args.test == True:
        print (args)
        print ('data_dir: ' + data_dir)
        print('save_dir: ' + save_dir)
        print( enc_filters)
    else:
        register(args.saveimg, args.savewarp, args.savevisual, args.num, args.weeknr, data_dir, args.atlasdir, save_dir, loaded_model, enc_filters)

if __name__ == '__main__':
    args = parse_args()
    main(args)
