from multi_nonrigid.train import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Launch trainig')
    parser.add_argument('--datadir', type=str, default='/trinity/home/wbastiaansen/data/8_12_weken/affine_reg', help='directory containing the data for training, folders with mask and atlases')
    parser.add_argument('--atlasdir', type=str, default='/trinity/home/wbastiaansen/data/8_12_weken/atlas', help='directory containing the atlases')
    parser.add_argument('--experimentid', type=str, help='Experiment id')
    parser.add_argument('--numiter', type=int, default=300, help='the number of iterations for training')
    parser.add_argument('--outputdir', type=str, default='/data/scratch/wbastiaansen/output', help='directory to save output')
    parser.add_argument('--regparam', type=float, default=1, help='trade-off parameter in loss')
    parser.add_argument('--atlaslist', type=str, nargs='+', default='00001', help='list of id of used atlases')
    parser.add_argument('--loadmodel', type=str, default='None', help='model to preload for training')
    parser.add_argument('--m', type=int, default=1, help='Number of atlases for optimization')
    parser.add_argument('--enc', type=int, default=16, help='number of filters in first layer enc')
    parser.add_argument('--dec', type=int, default=2, help='number of layers on full resolution')
    parser.add_argument('--diffeo', type=bool, default=False, help='if set to true diffeomorphic model is trained')
    parser.add_argument('--intsteps', type=int, default=7, help='number of integration steps for diffeomorphic model')
    parser.add_argument('--slurmid', type=str, help='slurm id of the job')



    args = parser.parse_args()
    return args

def main(args):
    data_dir = args.datadir + '/train'
    mask_file = args.atlasdir + '/mask/mask_ev.npz'
    model_dir = args.outputdir + '/model_out_' + args.slurmid + args.experimentid

    enc = [args.enc, 2*args.enc, 2*args.enc, 2*args.enc]
    if args.dec == 2:
        dec = [2*args.enc,2*args.enc, 2*args.enc, 2*args.enc, 2*args.enc, args.enc, args.enc]
    else:
        dec = [2*args.enc, 2*args.enc, 2*args.enc, 2*args.enc, 2*args.enc]

    if args.loadmodel == 'None':
        load_model_file = None
    else:
        load_model_file = args.loadmodel

    train(args.atlasdir, data_dir, model_dir, args.regparam, args.numiter, mask_file, args.atlaslist, load_model_file, args.m, enc, dec, args.diffeo, args.intsteps)


    
if __name__ == '__main__':
    args = parse_args()
    main(args)
                         
