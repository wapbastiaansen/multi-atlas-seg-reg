from multi_affine.train import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Launch training')
    parser.add_argument('--datadir',type=str, help='directory containing the data for training, folders with mask and atlases')
    parser.add_argument('--atlasdir', type=str, help='directory containing the atlases')
    parser.add_argument('--slurmid', type=str, help='slurm id of the job')
    parser.add_argument('--experimentid', type=str, help='experiment id')
    parser.add_argument('--numiter', type=int, default=300, help='number of iterations for training')
    parser.add_argument('--stage', type=str, default='1', help='the number of the stage launched for training')
    parser.add_argument('--regparam', type=float, default='1', help='value of the regularization parameter')
    parser.add_argument('--loadmodel', type=str, default='None', help='model to preload for training')
    parser.add_argument('--m', type=int, default=1, help='Number of atlases for optimization')
    parser.add_argument('--atlaslist', type=str, nargs='+', default='00001', help='list id of used atlases')
    parser.add_argument('--flip', type=int, default=0, help='0: no flips applied as data augmentation, 1: flips applied')
    parser.add_argument('--rot90', type=int, default=0, help='0: no 90 degree rotation are applied, 1: 90 degree rotations')
    parser.add_argument('--outputdir', type=str, help='directory to save output')
    parser.add_argument('--enc', type=int, default=64, help='number of filters in first layer enc')
    
    args = parser.parse_args()
    return args


def main(args):
    data_dir = args.datadir + '/train'
    mask_file = args.datadir + '/atlas/mask/mask_ev.npz'
    model_dir = args.outputdir + '/model_out_' + args.slurmid + args.experimentid

    if args.stage == '2':
        learning_rate = 0.00001
    else:
        learning_rate = 0.0001

    enc_global = [args.enc, 2*args.enc, 2*args.enc, 2*args.enc]

    if args.loadmodel == 'None':
        load_model_file = None
    else:
        load_model_file = args.loadmodel

    data_aug = [args.flip, args.rot90]
    
    if args.test == True:
        print(args)
        print('data_dir: ' + data_dir)
        print('model_dir: ' + model_dir)
        print('mask_file: ' + mask_file)
        print( enc_global)
        print(load_model_file)
        print(learning_rate)
        print(data_aug)
    else:
        train(args.atlasdir, data_dir, model_dir, args.regparam, args.stage, args.numiter, args.atlaslist, args.m, mask_file, enc_global, load_model_file, learning_rate, data_aug)

if __name__ == '__main__':
    args = parse_args()
    main(args)
