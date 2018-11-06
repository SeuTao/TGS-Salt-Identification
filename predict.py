import argparse
from data_process.data_loader import *
from data_process.transform import *
from loss.bce_losses import *
from loss.cyclic_lr import *
from loss.lovasz_losses import *
from utils import create_submission
from loss.metric import do_kaggle_metric
import time
import datetime

from train import SingleModelSolver
from utils import *

def votingAllCycle(root_dir , model_list, save_name):
    print('voting ensemble')

    def get_predict_dict(csv_list):
        # vote_thres = int(len(csv_list) / 2) + 1
        vote_dict = {}
        for csv in csv_list:
            csv_dict = decode_csv(csv_name=csv)
            for id in csv_dict:
                if id in vote_dict:
                    vote_dict[id] += csv_dict[id]
                else:
                    vote_dict[id] = csv_dict[id]
            csv_dict.clear()

        for id in vote_dict:
            vote_dict[id] = vote_dict[id] / (len(csv_list)*1.0)
        return vote_dict

    model_list = [os.path.join(root_dir, tmp) for tmp in model_list]

    vote_dict_list = []
    for dir_tmp in model_list:
        print(os.path.split(dir_tmp)[1])
        csv_list = create_csv_lists_recursive(dir_tmp)
        vote_dict_tmp = get_predict_dict(csv_list)
        vote_dict_list.append(vote_dict_tmp)


    vote_dict = {}
    for dict in vote_dict_list:
        for id in dict:
            if id in vote_dict:
                vote_dict[id] += dict[id] / len(vote_dict_list)
            else:
                vote_dict[id] = dict[id] / len(vote_dict_list)
        dict.clear()

    out = []
    for id in vote_dict:
        vote_dict[id][vote_dict[id] >= 0.5] = 1
        vote_dict[id][vote_dict[id] < 0.5] = 0
        vote_dict[id] = vote_dict[id].astype(np.uint8)
        out.append([id, vote_dict[id]])

    submission = create_submission(out)
    submission.to_csv(os.path.join(save_name), index=None)

    print('done')

def ensemble_models(root_dir ,model_list , save_name):
    votingAllCycle(root_dir, model_list, save_name)
    submission_apply_jigsaw_postprocessing(save_name)

def main(config):
    if config.mode == 'InferModel10Fold':
        solver = SingleModelSolver(config)
        for i in range(10):
            solver.infer_fold_all_Cycle(i)

    if config.mode == 'EnsembleModels':
        print(config.model_name_list)
        model_name_list = config.model_name_list.split(',')
        print(model_name_list)


        ensemble_models(config.model_save_path, model_name_list, config.save_sub_name)

    if config.mode == 'SolveJigsawPuzzles':
        from jigsaw.jigsaw_puzzles import solve_jigsaw_puzzles
        solve_jigsaw_puzzles(config.jigsaw_dir)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='InferModel10Fold', choices=['InferModel10Fold',
                                                                                 'EnsembleModels',
                                                                                 'SolveJigsawPuzzles'])

    parser.add_argument('--model_name_list', type=str, default= r'model_34', required = True)
    parser.add_argument('--save_sub_name', type=str, default= 'model_34_fold0.csv')
    parser.add_argument('--train_fold_index', type=int, default=0)
    parser.add_argument('--model', type=str, default='model_34')
    parser.add_argument('--model_name', type=str, default='model_34')

    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16*4)

    # Test settings
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--jigsaw_dir', type=str, default='./jigsaw')

    # no use
    parser.add_argument('--pseudo_csv', type=str, default = None)
    parser.add_argument('--pseudo_split', type=int, default = -1)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=20000)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cycle_num', type=int, default=7)
    parser.add_argument('--cycle_inter', type=int, default=50)
    parser.add_argument('--dice_bce_pretrain_epochs', type=int, default=10)
    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--bce_weight', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=16)

    config = parser.parse_args()
    print(config)
    main(config)



