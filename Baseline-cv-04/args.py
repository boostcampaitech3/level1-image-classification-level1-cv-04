import torch
import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for Mask Classification train/test')
    parser.add_argument('--model', default='efficientnet_b3') ### 
    parser.add_argument('--criterion', default='focal', help='[\'cross_entropy\', \'focal\', \'label_smoothing\', \'f1\']')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)') 
    parser.add_argument('--resize', type=int, nargs="+", default=(312, 312), help='Resize input image')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for data split')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--inference', type=bool, default=False, help='Inference single model')
    parser.add_argument('--earlystopping_patience', type=int, default=6)
    parser.add_argument('--scheduler_patience', type=int, default=2)
    parser.add_argument('--age_bound', type=int, default=59)
    parser.add_argument('--save_path', default=False, help='Trained model path')
    parser.add_argument('--num_classes', type=int, default=18)
    parser.add_argument('--oversampling', type=bool, default=False)
    parser.add_argument('--kfold', type=bool, default=False)
    parser.add_argument('--class_weights', type=bool, default=False)


    parse = parser.parse_args()
    params = {
        "MODEL": parse.model, 
        "CRITERION": parse.criterion, 
        "OPTIMIZER": parse.optimizer,
        "RESIZE": parse.resize, 
        "LEARNING_RATE": parse.learning_rate,
        "WEIGHT_DECAY": parse.weight_decay, 
        "BATCH_SIZE": parse.batch_size,
        "RANDOM_SEED": parse.random_seed,
        "DEVICE": parse.device,
        "INFERENCE": parse.inference,
        "EARLYSTOPPING_PATIENCE": parse.earlystopping_patience,
        "SCHEDULER_PATIENCE": parse.scheduler_patience,
        "AGE_BOUND": parse.age_bound,
        "SAVE_PATH": parse.save_path, 
        "NUM_CLASSES": parse.num_classes,
        "OVERSAMPLING": parse.oversampling,
        "KFOLD": parse.kfold,
        "CLASS_WEIGHTS": parse.class_weights
    }