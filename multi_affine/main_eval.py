import argparse
from multi_affine.eval import eval_affine

def parse_args():
    parser = argparse.ArgumentParser(description='Launch evaluation')
    parser.add_argument('--slurmid', type=str, help='slurm id of the job')
    parser.add_argument('--experimentid', type=str, help='experiment id')
    parser.add_argument('--evfile', type=str, help='excel file containign all the EV_VR')
    parser.add_argument('--atlasdir', type=str, help='directory where eligble atlas files can be found')
    parser.add_argument('--atlaslist', type=str, nargs='+', default='00001', help='list of id of used atlases')
    parser.add_argument('--segdir', type=str, help='directory with gt segmentations')
    parser.add_argument('--preprodir', type=str, help='directory containing the prepro .npz files (if applicable)')
    parser.add_argument('--set', type=str, default='val',  help='set of data for which we calculate Dice and ev error')
    parser.add_argument('--outputdir', type=str, help='directory to save output')
    parser.add_argument('--test', type=bool, default=False, help='variable for testing')
    parser.add_argument('--rerun', type=bool, default=False, help='if true: re run with other slurm id then run from which we retrieve data')
    parser.add_argument('--m', type=int, default=1, help='The number of atlases used for evaluation')
    parser.add_argument('--saveseg', type=bool, default=False, help='if true the segmenations are saved')
    args = parser.parse_args()

    return args

def main(args):
    
    if args.rerun == False:
        data_file = args.outputdir + '/model_out_' + args.slurmid + args.experimentid + '/warp_' + args.set +'/outcome.xlsx'
        warp_dir = args.outputdir + '/model_out_' + args.slurmid + args.experimentid + '/warp_' + args.set
    else:
        data_file = args.outputdir + '/warp_' +args.set + '/outcome.xlsx'
        warp_dir = args.outputdir + '/warp_' + args.set 

    if args.test==False:
        Dice_affine, EV_affine = eval_affine(data_file, args.evfile, args.atlasdir, warp_dir, args.segdir, args.preprodir, args.atlaslist, args.set, args.m, args.saveseg)
    else:
        print(args)
        print('data_file: ' + data_file)
        print('warp_dir: ' + warp_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)


